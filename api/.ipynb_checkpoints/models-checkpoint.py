from django.db import models


class Photo(models.Model):
    title = models.CharField(max_length=100)  # this field does not use in your project
    img = models.ImageField(upload_to='img/')

    def __str__(self):
        return self.title
