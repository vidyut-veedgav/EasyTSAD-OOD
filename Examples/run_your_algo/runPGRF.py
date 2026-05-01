from EasyTSAD.Controller import TSADController
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

if __name__ == "__main__":
    from EasyTSAD.Methods.PGRF import PGRF

    gctrl = TSADController()

    gctrl.set_dataset(
        dataset_type="MTS",
        dirname=r"d:\Sreya\Case_Western\OODResearch\EasyTSAD-OOD\datasets",
        datasets=["SMD"],
    )

    training_schema = "naive"
    method = "PGRF"

    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        hparams={
            "batch_size": 32,
            "window": 64,
            "epochs": 20,
            "lr": 1e-3,
            "num_protos": 8,
            "num_context_protos": 8,
            "num_spike_protos": 8,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_ff": 256,
            "top_m_percent": 20,
        },
        preprocess="z-score",
    )

    gctrl.set_evals([PointF1PA(), EventF1PA(), EventF1PA(mode="squeeze")])
    gctrl.do_evals(method=method, training_schema=training_schema)
    gctrl.plots(method=method, training_schema=training_schema)
