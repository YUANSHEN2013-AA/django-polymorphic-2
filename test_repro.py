import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'polymorphic.tests.settings')
django.setup()

from polymorphic.models import PolymorphicModel
from django.db import models

class Level1(PolymorphicModel):
    name = models.CharField(max_length=10)

class Level2(Level1):
    level2_name = models.CharField(max_length=10)

class Level3(Level2):
    level3_name = models.CharField(max_length=10)
