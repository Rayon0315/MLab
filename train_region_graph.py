"""Reuse canonical train.py with the official NAMLab partition dataset.

Only the dataset class is replaced. The standard SODLoss and the canonical
training loop remain unchanged because RegionGraph has no auxiliary loss.
"""
import train as train_module

from data.nrtr_dataset import NRTRSODDataset


def main():
    train_module.SODDataset = NRTRSODDataset
    train_module.main()


if __name__ == "__main__":
    main()
