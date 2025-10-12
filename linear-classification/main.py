from sklearn.datasets import load_breast_cancer
import numpy as np
from typing import cast
import streamlit as st

class LinearClassification:

    @property
    def X(self) -> np.ndarray:
        return self.__X

    @property
    def y(self) -> np.ndarray:
        return self.__y
    
    def __init__(self) -> None:
        # force sklearn to return X, y tuple for static typing
        X, y = cast(tuple[np.ndarray, np.ndarray], load_breast_cancer(return_X_y=True))

        self.__X: np.ndarray = X
        self.__y: np.ndarray = y

if __name__ == "__main__":
    st.set_page_config(page_title="Linear Classification", layout="wide")
    model = LinearClassification()
    st.title("Linear Classification")
    features, target = st.tabs(["Features", "Target"])
    with features:
        st.write("## Features")
        st.dataframe(model.X)
    with target:
        st.write("## Target")
        st.dataframe(model.y)