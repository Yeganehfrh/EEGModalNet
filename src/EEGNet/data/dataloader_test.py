from .datamodule port EEGNetDataModule

def test_init_datamodule():
    obj = EEGNetDataModule()

    assert object is not None
