"""
The original paper for the code below:
- GitHub: https://github.com/hezgit/TDM

- Citation:
@inproceedings{zhao2023transformed,
  title={Transformed Distribution Matching for Missing Value Imputation},
  author={Zhao, He and Sun, Ke and Dezfouli, Amir and Bonilla, Edwin V},
  booktitle={International Conference on Machine Learning},
  pages={42159--42186},
  year={2023},
  organization={PMLR}
}
"""
import ot
import torch
import logging
import numpy as np


class TDMImputer:
    def __init__(self, projector, im_lr=1e-2, proj_lr=1e-2, opt=torch.optim.RMSprop, niter=2000, batchsize=128,
                 n_pairs=1, noise=0.1):
        self.projector = projector
        self.im_lr = im_lr
        self.proj_lr = proj_lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.is_fitted = False

    def fit(self, X_train, verbose=True, report_interval=500):
        """
        Trains the TDM imputer using the training dataset.

        Parameters:
        X_train: torch.Tensor - The training dataset with missing values.
        verbose: bool - Whether to print training progress.
        report_interval: int - Interval for logging progress.
        """
        X_train = X_train.clone()
        n, d = X_train.shape

        mask = torch.isnan(X_train).double()
        torch.autograd.set_detect_anomaly(True)

        imps = (self.noise * torch.randn(mask.shape) + torch.nanmean(X_train, 0))[mask.bool()]
        imps.requires_grad = True

        im_optimizer = self.opt([imps], lr=self.im_lr)
        proj_optimizer = self.opt(self.projector.parameters(), lr=self.proj_lr)

        for i in range(self.niter):
            X_filled = X_train.detach().clone()
            X_filled[mask.bool()] = imps

            im_loss = 0
            proj_loss = 0

            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]

                X1_proj, _ = self.projector(X1)
                X2_proj, _ = self.projector(X2)

                M_proj = torch.cdist(X1_proj, X2_proj, p=2)
                a1 = torch.ones(X1.shape[0]) / X1.shape[0]
                a2 = torch.ones(X2.shape[0]) / X2.shape[0]
                a1.requires_grad = False
                a2.requires_grad = False

                ot_proj = ot.emd2(a1, a2, M_proj)
                im_loss += ot_proj
                proj_loss += ot_proj

            im_optimizer.zero_grad()
            im_loss.backward(retain_graph=True)
            im_optimizer.step()

            proj_optimizer.zero_grad()
            proj_loss.backward()
            proj_optimizer.step()

            if verbose and i % report_interval == 0:
                logging.info(f"Fitting Iteration {i}: Im Loss: {im_loss.item():.4f}, Proj Loss: {proj_loss.item():.4f}")

        self.fitted_imps = imps.detach().clone()
        self.is_fitted = True

    def transform(self, X_test, verbose=True, report_interval=500):
        """
        Imputes missing values in the test dataset using the trained TDM model.

        Parameters:
        X_test: torch.Tensor - The test dataset with missing values.

        Returns:
        torch.Tensor - The test dataset with imputed values.
        """
        if not self.is_fitted:
            raise RuntimeError("The TDMImputer must be fitted before calling transform.")

        X_test = X_test.clone()
        n, d = X_test.shape

        batch_size = self.batchsize
        if batch_size > n // 2:
            e = int(np.log2(n // 2))
            batch_size = 2 ** e

        mask = torch.isnan(X_test).double()
        torch.autograd.set_detect_anomaly(True)

        imps = (self.noise * torch.randn(mask.shape) + torch.nanmean(X_test, 0))[mask.bool()]
        imps.requires_grad = True
        im_optimizer = self.opt([imps], lr=self.im_lr)

        for i in range(self.niter):
            X_filled = X_test.detach().clone()
            X_filled[mask.bool()] = imps

            im_loss = 0
            for _ in range(self.n_pairs):
                idx1 = np.random.choice(X_filled.shape[0], batch_size, replace=False)
                idx2 = np.random.choice(X_filled.shape[0], batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]

                X1_proj, _ = self.projector(X1)
                X2_proj, _ = self.projector(X2)

                M_proj = torch.cdist(X1_proj, X2_proj, p=2)
                a1 = torch.ones(X1.shape[0]) / X1.shape[0]
                a2 = torch.ones(X2.shape[0]) / X2.shape[0]
                a1.requires_grad = False
                a2.requires_grad = False

                im_loss += ot.emd2(a1, a2, M_proj)

            im_optimizer.zero_grad()
            im_loss.backward()
            im_optimizer.step()

            if verbose and i % report_interval == 0:
                logging.info(f"Transforming Iteration {i}: Im Loss: {im_loss.item():.4f}")

        X_filled = X_test.detach().clone()
        X_filled[mask.bool()] = imps
        return X_filled.detach()
