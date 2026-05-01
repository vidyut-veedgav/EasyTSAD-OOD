from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

if __name__ == "__main__":
    from EasyTSAD.Methods.CATCH import CATCH

    gctrl = TSADController()

    gctrl.set_dataset(
        dataset_type="MTS",
        dirname=r"d:\Sreya\Case_Western\OODResearch\EasyTSAD-OOD\datasets",
        datasets=["SMD"],
    )

    training_schema = "naive"
    method = "CATCH"

    # CATCH applies its own StandardScaler internally, so use preprocess="raw"
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "seq_len": 192,
            "num_epochs": 3,
            "batch_size": 128,
            "patience": 3,
            "lr": 0.0001,
            "Mlr": 0.00001,
        },
        preprocess="raw",
    )

    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema=training_schema)
    gctrl.plots(method=method, training_schema=training_schema)
