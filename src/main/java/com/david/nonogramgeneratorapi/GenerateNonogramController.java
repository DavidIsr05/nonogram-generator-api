package com.david.nonogramgeneratorapi;

import com.david.nonogramgeneratorapi.dtos.*;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
@RequestMapping("api/v1/generate-nonogram")
public class GenerateNonogramController {

    private final GenerateNonogramService generateNonogramService;

    public GenerateNonogramController(GenerateNonogramService generateNonogramService) {
        this.generateNonogramService = generateNonogramService;
    }

    @PostMapping
    public nonogramResponseDto generateNonogram(@RequestBody nonogramGenerationRequestDto requestDto) throws IOException {
        return generateNonogramService.generateNonogram(requestDto);
    }
}