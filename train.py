import cam_image
import classifier

# cam = cam_image.frames()
model = classifier.classification()
model.create_classification_model()
model.data_generators()
model.finalise()


