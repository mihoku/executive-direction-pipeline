from django.db import models
from django.utils import timezone

class Unit(models.Model):
    """
    Represents an organizational unit.
    e.g., "DJP", "DJBC", "BATII"
    """
    name = models.CharField(max_length=255)
    abbreviation = models.CharField(max_length=50, unique=True, db_index=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class MinutesOfMeeting(models.Model):
    """
    Stores the executive minutes of meetings.
    """
    title = models.CharField(max_length=255)
    text = models.TextField()
    summary = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    meeting_date = models.DateField(default=timezone.now)
    original_audio_filename = models.CharField(max_length=255, blank=True, null=True)
    original_pdf_filename = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-created_at']

class TaskAssignment(models.Model):
    """
    Stores a single task extracted from a MoM, assigned to one or more units.
    """
    mom = models.ForeignKey(MinutesOfMeeting, on_delete=models.CASCADE, related_name='tasks')
    description = models.TextField()
    assigned_units = models.ManyToManyField(Unit, related_name='tasks', blank=True)
    verbatim_sentence = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description[:50]
    
    class Meta:
        ordering = ['created_at']
