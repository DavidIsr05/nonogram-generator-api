package com.david.nonogramgeneratorapi;

import com.david.nonogramgeneratorapi.dtos.*;
import org.apache.commons.io.FileUtils;
import org.imgscalr.Scalr;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Base64;

@Service
public class GenerateNonogramService {
    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto body) throws IOException {
        String outputPath = "src/main/resources/";

        byte[] decodedBytes = Base64.getDecoder().decode(body.getImageBase64());
        File originalImageFile = new File(outputPath + "original-image.jpg");
        FileUtils.writeByteArrayToFile(originalImageFile, decodedBytes);

        BufferedImage originalBufferedImage = ImageIO.read(originalImageFile);

        BufferedImage scaledBufferedImage = Scalr.resize(originalBufferedImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT,
                calculateImageSizeForScalingBasedOnDifficulty(body.getDifficulty()), Scalr.OP_ANTIALIAS);

        BufferedImage grayScaleBufferImage = new BufferedImage(scaledBufferedImage.getWidth(), scaledBufferedImage.getHeight(),
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics = grayScaleBufferImage.getGraphics();
        graphics.drawImage(scaledBufferedImage, 0, 0, null);
        graphics.dispose();

        long totalBrightness = calculateTotalBrightnessOfImage(grayScaleBufferImage);
        int pixelCount = grayScaleBufferImage.getWidth() * grayScaleBufferImage.getHeight();

        int threshold = (int) (totalBrightness / pixelCount) + body.getContrast();

        BufferedImage blackAndWhiteBufferedImage = generateBlackAndWhiteImage(grayScaleBufferImage, threshold);

        File blackAndWhiteImageFile = new File(outputPath + "/black-and-white-image.png");
        ImageIO.write(blackAndWhiteBufferedImage, "png", blackAndWhiteImageFile);

        boolean[][] nonogram = generateNonogram(blackAndWhiteBufferedImage);

        if (originalImageFile.delete()) {
            System.out.println("Deleted the file: " + originalImageFile.getName());
        } else {
            System.out.println("Failed to delete the file");
        }

        byte[] fileContent = FileUtils.readFileToByteArray(new File(blackAndWhiteImageFile.getPath()));
        String encodedString = Base64.getEncoder().encodeToString(fileContent);

        if (blackAndWhiteImageFile.delete()) {
            System.out.println("Deleted the file: " + blackAndWhiteImageFile.getName());
        } else {
            System.out.println("Failed to delete the file");
        }

        return new nonogramResponseDto(nonogram, encodedString);
    }

    private int calculateImageSizeForScalingBasedOnDifficulty(Difficulty difficulty){
        return 20 + (10 * difficulty.ordinal());
    }

    private int calculatePixelBrightness(BufferedImage source, int imageYIndex, int imageXIndex) {
        int rgb = source.getRGB(imageXIndex, imageYIndex);

        Color pixelColorRGB = new Color(rgb);

        return pixelColorRGB.getRed();
    }

    private int calculateTotalBrightnessOfImage(BufferedImage grayScaleBufferImage){
        int totalBrightness = 0;
        for (int imageYIndex = 0; imageYIndex < grayScaleBufferImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < grayScaleBufferImage.getWidth(); imageXIndex++) {
                int brightness = calculatePixelBrightness(grayScaleBufferImage, imageYIndex, imageXIndex);
                totalBrightness += brightness;
            }
        }

        return totalBrightness;
    }

    private BufferedImage generateBlackAndWhiteImage(BufferedImage grayScaleBufferImage, int threshold){
        BufferedImage blackAndWhiteImage = new BufferedImage(
                grayScaleBufferImage.getWidth(),
                grayScaleBufferImage.getHeight(),
                BufferedImage.TYPE_BYTE_BINARY
        );

        int whiteColor = 0xFFFFFF;
        int blackColor = 0x000000;

        for (int imageYIndex = 0; imageYIndex < grayScaleBufferImage.getHeight(); imageYIndex++) {
            for (int imageXIndex = 0; imageXIndex < grayScaleBufferImage.getWidth(); imageXIndex++) {
                int brightness = calculatePixelBrightness(grayScaleBufferImage, imageYIndex, imageXIndex);
                int newPixel = (brightness >= threshold) ? whiteColor : blackColor;
                blackAndWhiteImage.setRGB(imageXIndex, imageYIndex, newPixel);
            }
        }

        return blackAndWhiteImage;
    }

    private boolean[][] generateNonogram(BufferedImage blackAndWhiteImage){

        boolean[][] nonogram = new boolean[blackAndWhiteImage.getWidth()][blackAndWhiteImage.getHeight()];

        for (int imageXIndex = 0; imageXIndex < blackAndWhiteImage.getHeight(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < blackAndWhiteImage.getWidth(); imageYIndex++) {
                int rgb = blackAndWhiteImage.getRGB(imageXIndex, imageYIndex);
                Color color = new Color(rgb);

                boolean shouldPixelBeBlack = color.getRed() == 0 && color.getGreen() == 0 && color.getBlue() == 0;

                nonogram[imageXIndex][imageYIndex] = shouldPixelBeBlack;
            }
        }

        return nonogram;
    }
}