from django.db import models

# Create your models here.

class ImageModel(models.Model):
     Image = models.ImageField(upload_to='images/')
     @classmethod
     def create(cls, image):
        imageF = cls(Image = image)
        return imageF
     
           
