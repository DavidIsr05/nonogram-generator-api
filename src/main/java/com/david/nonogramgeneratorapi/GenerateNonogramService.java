package com.david.nonogramgeneratorapi;

import com.david.nonogramgeneratorapi.dtos.*;
import com.david.nonogramgeneratorapi.util.PixelUtils;
import jakarta.annotation.PostConstruct;
import org.imgscalr.Scalr;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.Color;
import java.awt.AlphaComposite;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ByteArrayOutputStream;
import java.util.Base64;

@Service
public class GenerateNonogramService {

    private Net u2netModel;
    final String MODEL_PATH = "src/main/resources/u2net.onnx";

    PixelUtils pixel = new PixelUtils();

    static {

        final File ALGORITHM_LIBRARY_FILE = new File("src/main/resources/libopencv_java4120.dylib");

        try {
            System.load(ALGORITHM_LIBRARY_FILE.getAbsolutePath());
        } catch (UnsatisfiedLinkError e) {
            throw new UnsatisfiedLinkError("Can't load openCV jar files. Error message: " + e);
        }
    }

    @PostConstruct
    public void initModel() {
        u2netModel = Dnn.readNetFromONNX(MODEL_PATH);
        if (u2netModel.empty()) {
            throw new RuntimeException(new CouldNotLoadModelException());
        }
    }

    public nonogramResponseDto generateNonogram(nonogramGenerationRequestDto requestBody) throws Exception {
        byte[] originalImageAsBytes = Base64.getDecoder().decode(requestBody.getImageBase64());

        ByteArrayInputStream originalImageAsByteArrayStream = new ByteArrayInputStream(originalImageAsBytes);

        BufferedImage originalImage = ImageIO.read(originalImageAsByteArrayStream);

        BufferedImage mainObjectFromModel = detectMainObject(originalImage);

        BufferedImage dimmedImage = applyDimFactor(originalImage, mainObjectFromModel, requestBody.getMainObjectDimFactor());

        int matrixSize = requestBody.getDifficulty().getMatrixSize();

        BufferedImage downscaledDimmedImage = Scalr.resize(dimmedImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage downscaledOriginalImage = Scalr.resize(originalImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, matrixSize, Scalr.OP_ANTIALIAS);

        BufferedImage grayScaledImage = new BufferedImage(matrixSize, matrixSize, BufferedImage.TYPE_BYTE_GRAY);

        Graphics graphics = grayScaledImage.getGraphics();

        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, matrixSize, matrixSize);
        graphics.drawImage(downscaledDimmedImage, 0, 0, null);
        graphics.dispose();

        int threshold = pixel.calculateAverageBrightness(downscaledOriginalImage, matrixSize);

        boolean[][] nonogram = generateNonogram(grayScaledImage, threshold);

        BufferedImage downscaledOriginalImageForPreview = originalImage;

        final int SMALL_IMAGE_RESOLUTION = 500;

        if (originalImage.getHeight() > SMALL_IMAGE_RESOLUTION | originalImage.getWidth() > SMALL_IMAGE_RESOLUTION) {
            downscaledOriginalImageForPreview = Scalr.resize(originalImage, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.AUTOMATIC, SMALL_IMAGE_RESOLUTION, Scalr.OP_ANTIALIAS);
        }

        BufferedImage previewImage = createPreview(nonogram, downscaledOriginalImageForPreview, threshold, requestBody.getPreviewImageIntRGB());

        String previewImageBase64 = bufferedImageToBase64(previewImage);

        String downscaledOriginalImageForCompletedNonogramsBase64 = bufferedImageToBase64(downscaledOriginalImage);

        return new nonogramResponseDto(nonogram, previewImageBase64, downscaledOriginalImageForCompletedNonogramsBase64, requestBody.getMainObjectDimFactor(), requestBody.getDifficulty());
    }

    private BufferedImage detectMainObject(BufferedImage inputImage) throws Exception {
        Mat inputImageInMatFormat = bufferedImageToMat(inputImage);

        if (inputImageInMatFormat.empty()) {
            throw new FileNotFoundException("Problem while loading original image for model in: 'detectMainObjectUsingModel'");
        }
        Mat mainObjectFromModel = Dnn.blobFromImage(inputImageInMatFormat, 0.01, new Size(250, 250), new Scalar(0, 0, 0), true, false);
        u2netModel.setInput(mainObjectFromModel);

        Mat originalMatOfMainObject = u2netModel.forward();

        Mat reshapedMatOfMainObject = originalMatOfMainObject.reshape(1, 250);

        Mat resizedMatOfMainObjectBasedOnOriginalInputImage = new Mat();

        Imgproc.resize(reshapedMatOfMainObject, resizedMatOfMainObjectBasedOnOriginalInputImage, inputImageInMatFormat.size());

        Mat binaryMatOfMainObject = new Mat();

        final double binaryImageCreationThreshold = 0.5;

        Imgproc.threshold(resizedMatOfMainObjectBasedOnOriginalInputImage, binaryMatOfMainObject, binaryImageCreationThreshold, 1, Imgproc.THRESH_BINARY);

        binaryMatOfMainObject.convertTo(binaryMatOfMainObject, CvType.CV_8U, 255);

        return matToBufferedImage(binaryMatOfMainObject);
    }

    private BufferedImage applyDimFactor(BufferedImage originalImage, BufferedImage mainObjectFromModel, double dimFactor) {
        BufferedImage duplicatedOriginalImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());
        Graphics graphics = duplicatedOriginalImage.getGraphics();
        graphics.drawImage(originalImage, 0, 0, null);
        graphics.dispose();

        for (int imageXIndex = 0; imageXIndex < mainObjectFromModel.getWidth(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < mainObjectFromModel.getHeight(); imageYIndex++) {
                boolean isMainObjectPixel = pixel.calculatePixelBrightness(mainObjectFromModel, imageXIndex, imageYIndex) != 0;
                int originalPixel = duplicatedOriginalImage.getRGB(imageXIndex, imageYIndex);

                int updatedPixel = pixel.getUpdatedPixel(originalPixel, isMainObjectPixel, dimFactor);

                duplicatedOriginalImage.setRGB(imageXIndex, imageYIndex, updatedPixel);
            }
        }

        return duplicatedOriginalImage;
    }

    private boolean[][] generateNonogram(BufferedImage grayScaledImage, int threshold) {
        boolean[][] nonogram = new boolean[grayScaledImage.getWidth()][grayScaledImage.getHeight()];

        for (int imageXIndex = 0; imageXIndex < grayScaledImage.getWidth(); imageXIndex++) {
            for (int imageYIndex = 0; imageYIndex < grayScaledImage.getHeight(); imageYIndex++) {

                int brightness = pixel.calculatePixelBrightness(grayScaledImage, imageXIndex, imageYIndex);
                boolean isPixelBlack = brightness < threshold;

                nonogram[imageXIndex][imageYIndex] = isPixelBlack;
            }
        }

        return nonogram;
    }

    private BufferedImage createPreview(boolean[][] nonogram, BufferedImage originalImage, int threshold, int previewImageIntRGB) {
        int pixelWidthRatio = originalImage.getWidth() / nonogram.length;
        int pixelHeightRatio = originalImage.getHeight() / nonogram.length;

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        Color highlightColor = new Color(previewImageIntRGB);

        BufferedImage previewImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

        Graphics2D graphics2d = previewImage.createGraphics();
        graphics2d.setComposite(AlphaComposite.Clear);
        graphics2d.fillRect(0, 0, width, height);

        final float highDimFactor = 0.2f;
        final float lowDimFactor = 0.6f;

        final int blackAndWhitePixelThreshold = 128;

        for (int nonogramXIndex = 0; nonogramXIndex < nonogram.length; nonogramXIndex++) {
            for (int nonogramYIndex = 0; nonogramYIndex < nonogram[0].length; nonogramYIndex++) {

                if (nonogram[nonogramXIndex][nonogramYIndex]) {

                    int coordinateXOnOriginalBasedOnNonogram = nonogramXIndex * pixelWidthRatio;
                    int coordinateYOnOriginalBasedOnNonogram = nonogramYIndex * pixelHeightRatio;

                    float previewOpacity = threshold < blackAndWhitePixelThreshold ? highDimFactor : lowDimFactor;

                    graphics2d.setColor(highlightColor);
                    graphics2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, previewOpacity));
                    graphics2d.drawOval(coordinateXOnOriginalBasedOnNonogram, coordinateYOnOriginalBasedOnNonogram, pixelWidthRatio, pixelHeightRatio);
                }
            }
        }
        graphics2d.dispose();

        return previewImage;
    }

    public static Mat bufferedImageToMat(BufferedImage inputImage) {
        BufferedImage duplicatedImage = new BufferedImage(inputImage.getWidth(), inputImage.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        duplicatedImage.getGraphics().drawImage(inputImage, 0, 0, null);

        byte[] duplicatedImageInBytes = ((DataBufferByte) duplicatedImage.getRaster().getDataBuffer()).getData();

        Mat matFormatImage = new Mat(duplicatedImage.getHeight(), duplicatedImage.getWidth(), CvType.CV_8UC3);
        matFormatImage.put(0, 0, duplicatedImageInBytes);

        return matFormatImage;
    }

    public static BufferedImage matToBufferedImage(Mat inputMatImage) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (inputMatImage.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }

        BufferedImage bufferedImageFormatImage = new BufferedImage(inputMatImage.cols(), inputMatImage.rows(), type);

        byte[] inputImageInBytes = ((DataBufferByte) bufferedImageFormatImage.getRaster().getDataBuffer()).getData();

        inputMatImage.get(0, 0, inputImageInBytes);

        return bufferedImageFormatImage;
    }

    public static String bufferedImageToBase64(BufferedImage inputImage) throws IOException {
        ByteArrayOutputStream inputImageInByteArray = new ByteArrayOutputStream();

        ImageIO.write(inputImage, "png", inputImageInByteArray);

        byte[] imageBytes = inputImageInByteArray.toByteArray();

        return Base64.getEncoder().encodeToString(imageBytes);
    }
}