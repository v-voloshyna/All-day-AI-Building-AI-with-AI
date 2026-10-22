import argparse
import joblib

_model = joblib.load("model.joblib")


def predict(text: str) -> str:
    return str(_model.predict([text])[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    print(predict(ap.parse_args().text))
