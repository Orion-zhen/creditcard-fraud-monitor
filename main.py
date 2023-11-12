import pandas as pd
import numpy as np
from utils.args import parser
from utils.dataset import CreditCardDataset, splitter, decimation
from utils.model import Classifier
from training.dl import train_dl, predict


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.decimation:
        decimation()
        data = pd.read_csv("creditcard-decimation.csv")
    else:
        data = pd.read_csv("creditcard.csv")
    
    full_dataset = CreditCardDataset(data)
    train_dataset, test_dataset = splitter(full_dataset, args.split)
    input_size = len(full_dataset.features)
    model = Classifier(input_size=input_size)
    if args.device == "xpu" or args.device == "ipex":
        try:
            import intel_extension_for_pytorch as ipex

            model = ipex.optimize(model)
        except:
            raise ModuleNotFoundError("Please install ipex first")

    best_loss = train_dl(
        model=model,
        device=args.device,
        train_dataset=train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path,
    )

    accuracy = predict(
        model=model,
        device=args.device,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
    print(f"Best loss: {best_loss}, Accuracy: {accuracy}")