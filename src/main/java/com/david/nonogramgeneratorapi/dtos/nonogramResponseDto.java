package com.david.nonogramgeneratorapi.dtos;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class nonogramResponseDto implements Serializable {
    private boolean[][] nonogram;
    private String blackAndWhiteImageBase64;
}
