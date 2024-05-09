from .datamodule import LEMONEEGDataModule


def test_init_datamodule():
    obj = LEMONEEGDataModule()

    assert obj is not None
