def eval():
    from ultralytics import YOLO, RTDETR
    model = YOLO('./runs/detect/.../weights/best.pt')
    val_args = {
        'data' : 'dataset.yaml',
        'split' : 'test'
    }
    result = model.val(**val_args)
    print("☢️ model.info", model.model.info())
    
if __name__ == '__main__':
    eval()
