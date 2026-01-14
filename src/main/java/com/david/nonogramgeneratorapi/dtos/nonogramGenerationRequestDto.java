package com.david.nonogramgeneratorapi.dtos;

import com.david.nonogramgeneratorapi.Colors;
import com.david.nonogramgeneratorapi.Difficulty;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class nonogramGenerationRequestDto implements Serializable {
    private String imageBase64;
    private Difficulty difficulty;
    private double pixelHighlightValue;
    private Colors previewImageHighlightColor;
}