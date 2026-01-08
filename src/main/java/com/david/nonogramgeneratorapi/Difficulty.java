package com.david.nonogramgeneratorapi;

public enum Difficulty {
    EASY(20),
    MEDIUM(30),
    HARD(40);

    public final int matrixSize;

    Difficulty(int matrixSize){
        this.matrixSize = matrixSize;
    }
}
