from django.contrib import admin
from .models import Unit, MinutesOfMeeting, TaskAssignment

@admin.register(Unit)
class UnitAdmin(admin.ModelAdmin):
    list_display = ('name', 'abbreviation', 'description') # Columns to show in the list view
    search_fields = ('name', 'abbreviation') # Enable searching by these fields

@admin.register(MinutesOfMeeting)
class MinutesOfMeetingAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('title', 'text', 'summary')

@admin.register(TaskAssignment)
class TaskAssignmentAdmin(admin.ModelAdmin):
    list_display = ('description_short', 'mom', 'assigned_unit_list', 'created_at')
    list_filter = ('mom', 'assigned_units', 'created_at')
    search_fields = ('description',)

    def description_short(self, obj):
        return obj.description[:80] + '...' if len(obj.description) > 80 else obj.description
    description_short.short_description = 'Description'

    def assigned_unit_list(self, obj):
        return ", ".join([unit.abbreviation for unit in obj.assigned_units.all()])
    assigned_unit_list.short_description = 'Assigned Units'