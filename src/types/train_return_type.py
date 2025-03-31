from enum import Enum

class TrainReturnType(Enum):
    SCORE = "score"
    VAL_LOSS = "val_loss"
    TEST_LOSS = "test_loss"
