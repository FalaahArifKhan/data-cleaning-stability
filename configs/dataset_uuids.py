from configs.constants import (GERMAN_CREDIT_DATASET, BANK_MARKETING_DATASET, CARDIOVASCULAR_DISEASE_DATASET,
                               DIABETES_DATASET, LAW_SCHOOL_DATASET, ACS_INCOME_DATASET, ErrorRepairMethod)


DATASET_UUIDS = {
    ACS_INCOME_DATASET: {
        ErrorRepairMethod.deletion.value: '1ac5f3fc-ee4c-4e7d-9ce3-58c177ff966f',
        ErrorRepairMethod.median_mode.value: '2b2997c4-2b77-409e-b23f-ff8c6e2f81a6',
        ErrorRepairMethod.median_dummy.value: 'c56c5770-7aae-4933-a527-148e8255c54f',
        ErrorRepairMethod.miss_forest.value: '182dd22b-1f81-4cc2-aa43-bca6874c0183',
        ErrorRepairMethod.k_means_clustering.value: 'de1bc232-12c9-4964-9299-e6d2c80663f3',
        ErrorRepairMethod.datawig.value: '225673a3-700a-433a-88eb-90e1cbd42905',
        ErrorRepairMethod.discriminative_dl.value: '29de4990-f4c1-4041-a73f-388742fd07d7',
        ErrorRepairMethod.boost_clean.value: '4acc098f-98d4-4a61-9a67-e2c190de8ddf',
        ErrorRepairMethod.cp_clean.value: 'cec0130c-c5e7-4297-ab5d-4ba5205e7a73',
    }
}
