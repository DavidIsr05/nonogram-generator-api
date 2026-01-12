package com.david.nonogramgeneratorapi;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public enum Difficulty {
    EASY(20),
    MEDIUM(30),
    HARD(40);

    private final int matrixSize;
}
