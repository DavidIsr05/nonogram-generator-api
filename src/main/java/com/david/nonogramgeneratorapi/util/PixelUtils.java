package com.david.nonogramgeneratorapi.util;

import java.awt.*;
import java.awt.image.BufferedImage;

public class PixelUtils {
    public int getUpdatedPixelColor(int RGBColor, double dimFactor){
        final int maxPixelColorValue = 255;

        return Math.min(maxPixelColorValue, Math.max(0, (int) (RGBColor * dimFactor)));
    }

    public int calculatePixelBrightness(BufferedImage inputImage, int imageXIndex, int imageYIndex) {
        int rgb = inputImage.getRGB(imageXIndex, imageYIndex);
        Color pixelColorRGB = new Color(rgb);

        return (pixelColorRGB.getRed() + pixelColorRGB.getGreen() + pixelColorRGB.getBlue()) / 3;
    }

    public int getUpdatedPixel(int originalPixel, boolean isMainObjectPixel, double dimFactor) {
        if (isMainObjectPixel) {
            Color color = new Color(originalPixel, true);

            final int newRed = getUpdatedPixelColor(color.getRed(), dimFactor);
            final int newGreen = getUpdatedPixelColor(color.getGreen(), dimFactor);
            final int newBlue = getUpdatedPixelColor(color.getBlue(), dimFactor);
            final int alpha = color.getAlpha();

            Color dimmedColor = new Color(newRed, newGreen, newBlue, alpha);
            originalPixel = dimmedColor.getRGB();
        }

        return originalPixel;
    }

    public int calculateAverageBrightness(BufferedImage grayScaledImage, int matrixSize) {
        long totalBrightness = 0;

        for (int imageXIndex = 0; imageXIndex < grayScaledImage.getHeight(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < grayScaledImage.getWidth(); imageYIndex++) {
                totalBrightness += calculatePixelBrightness(grayScaledImage, imageXIndex, imageYIndex);
            }
        }

        int pixelCount = matrixSize * matrixSize;

        return (int) (totalBrightness / pixelCount);
    }
}
