import eyeprocesspy as ep

spec = ep.irt_model_spec(
    id="accuracy_time_gaze_demo",
    latent=["ability", "speed", "engagement"],
    channels={
        "response": ep.irt_response_channel("2pl"),
        "rt": ep.irt_rt_channel("lognormal"),
        "gaze": ep.irt_count_channel("poisson"),
    },
)
assert spec.status == "experimental"
models = ep.list_irt_models()
assert "joint_gaze_rt" in set(models["id"])
