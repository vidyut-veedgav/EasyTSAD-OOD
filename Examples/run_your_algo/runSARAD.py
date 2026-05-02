from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

if __name__ == "__main__":
    from EasyTSAD.Methods.SARAD import SARAD

    gctrl = TSADController()

    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="/home/sks190/EasyTSAD-OOD/datasets",
        datasets=["SMD"],
    )

    training_schema = "naive"
    method = "SARAD"

    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "batch_size": 32,
            "window": 100,
            "epochs": 10,
            "lr": 7e-4,
            "model_size": 64,
            "num_layers": 3,
            "num_heads": 8,
            "num_patches": 2,
            "detector_size": 64,
            "dropout": 0.1,
            "detec_weight": 100.0,
        },
        preprocess="z-score",
    )

    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema=training_schema)
    gctrl.plots(method=method, training_schema=training_schema)
