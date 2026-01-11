package com.david.nonogramgeneratorapi;

public class CouldNotLoadModelException extends Exception{
    CouldNotLoadModelException(String message){
        super((message));
    }
}
