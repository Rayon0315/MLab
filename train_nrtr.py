"""Reuse train.py with NRTR dataset/loss."""
import train as train_module
from data.nrtr_dataset import NRTRSODDataset
from losses.nrtr_sod_loss import NRTRSODLoss

def main():
    train_module.SODDataset = NRTRSODDataset
    train_module.SODLoss = NRTRSODLoss
    train_module.main()

if __name__ == '__main__':
    main()
