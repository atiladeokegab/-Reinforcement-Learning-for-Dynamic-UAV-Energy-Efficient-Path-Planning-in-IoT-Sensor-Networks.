FROM ubuntu:latest
LABEL authors="okeat"

ENTRYPOINT ["top", "-b"]