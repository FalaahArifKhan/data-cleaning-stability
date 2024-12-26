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

    def fit_transform(self, X_train, X_tests=None, verbose=True, report_interval=500):
        """
        Trains the TDM imputer using the training dataset.

        Parameters:
        X_train: torch.Tensor - The training dataset with missing values.
        X_tests: list of torch.Tensor - Optional list of test datasets.
        verbose: bool - Whether to print training progress.
        report_interval: int - Interval for logging progress.
        """
        X_train = X_train.clone()
        n, d = X_train.shape

        batch_size = self.batchsize
        if batch_size > n // 2:
            e = int(np.log2(n // 2))
            batch_size = 2 ** e

        mask_train = torch.isnan(X_train).double()
        torch.autograd.set_detect_anomaly(True)

        # Initialize imps for training data
        imps_train = (self.noise * torch.randn(mask_train.shape) + torch.nanmean(X_train, 0))[mask_train.bool()]
        imps_train.requires_grad = True

        # Initialize imps for each test dataset
        imps_tests = []
        mask_tests = []
        if X_tests is not None:
            for X_test in X_tests:
                X_test = X_test.clone()
                mask_test = torch.isnan(X_test).double()
                imps_test = (self.noise * torch.randn(mask_test.shape) + torch.nanmean(X_test, 0))[mask_test.bool()]
                imps_test.requires_grad = True
                imps_tests.append(imps_test)
                mask_tests.append(mask_test)

        # Combine all imps for optimization
        all_imps = [imps_train] + imps_tests

        # Optimizers
        im_optimizer = self.opt(all_imps, lr=self.im_lr)
        proj_optimizer = self.opt(self.projector.parameters(), lr=self.proj_lr)

        for i in range(self.niter):
            X_filled = X_train.detach().clone()
            X_filled[mask_train.bool()] = imps_train

            im_loss = 0
            proj_loss = 0

            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, batch_size, replace=False)
                idx2 = np.random.choice(n, batch_size, replace=False)

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

            if torch.isnan(im_loss).any() or torch.isinf(im_loss).any():
                logging.info("im_loss Nan or inf loss")
                break

            if torch.isnan(proj_loss).any() or torch.isinf(proj_loss).any():
                logging.info("proj_loss Nan or inf loss")
                break

            # Optimization steps
            im_optimizer.zero_grad()
            im_loss.backward(retain_graph=True)
            im_optimizer.step()

            proj_optimizer.zero_grad()
            proj_loss.backward()
            proj_optimizer.step()

            if verbose and i % report_interval == 0:
                logging.info(f"Fitting Iteration {i}: Im Loss: {im_loss.item():.4f}, Proj Loss: {proj_loss.item():.4f}")

        imps_train, imps_tests = all_imps[0], all_imps[1:]
        X_train_imp = X_train.detach().clone()
        X_train_imp[mask_train.bool()] = imps_train

        X_tests_imp = []
        if X_tests is not None:
            for idx, imps_test in enumerate(imps_tests):
                X_test = X_tests[idx]
                mask_test = mask_tests[idx]

                X_test_imp = X_test.detach().clone()
                X_test_imp[mask_test.bool()] = imps_test
                X_test_imp = X_test_imp.detach()
                X_tests_imp.append(X_test_imp)

        return X_train_imp.detach(), X_tests_imp
