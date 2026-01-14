package com.david.nonogramgeneratorapi;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.awt.*;

@AllArgsConstructor
@Getter
public enum Colors {
    RED(Color.RED),
    BLUE(Color.BLUE),
    PINK(Color.PINK),
    BLACK(Color.BLACK);
    //TODO add more colors
    private final Color color;
}
