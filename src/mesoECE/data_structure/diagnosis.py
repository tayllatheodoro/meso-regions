from enum import Enum


class Diagnosis(Enum):
    MALIGNANT = 1
    NON_MALIGNANT = 0


class SubclassMalignant(Enum):
    MDC = 0
    SARCOMATOID = 1
    EPITHELIOID = 2
    BIPHASIC= 3


class SubclassNonMalignant(Enum):
    BDC = 0
    BAPE = 1
