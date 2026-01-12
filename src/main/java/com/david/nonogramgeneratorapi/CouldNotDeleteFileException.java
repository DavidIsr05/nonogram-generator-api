package com.david.nonogramgeneratorapi;

import java.nio.file.FileSystemException;

public class CouldNotDeleteFileException extends FileSystemException {
    CouldNotDeleteFileException(String message){
        super((message));
    }
}
