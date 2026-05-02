from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

if __name__ == "__main__":
    from EasyTSAD.Methods.KANAD import KANAD

    gctrl = TSADController()

    # MTS datasets — SMD, MSL, SMAP, SWaT, PSM are all available
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="/home/sks190/EasyTSAD-OOD/datasets",
        datasets=["SMD"],
    )

    training_schema = "naive"
    method = "KANAD"

    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "batch_size": 64,
            "window": 64,
            "order": 4,
            "epochs": 20,
            "lr": 1e-3,
        },
        preprocess="z-score",
    )

    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema=training_schema)
    gctrl.plots(method=method, training_schema=training_schema)
