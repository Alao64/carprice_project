import joblib

for name in ["model.pkl", "preprocessor.pkl", "best_individual_model.pkl"]:
    path = f"artifacts/{name}"
    obj = joblib.load(path)
    joblib.dump(obj, path, compress=3)
    print(name, "done")