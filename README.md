### Build docker
```
$ docker-compose build
$ docker-compose up -d
$ docker exec -it <container_id> bash
```

### Convert darknet weights to tensorflow model
```
$ python save_model.py --model yolov4
```

### Run
```
$ python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4
```

### Reference
- [yolo4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
