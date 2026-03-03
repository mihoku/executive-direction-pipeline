"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include # Import include
from django.views.generic import TemplateView # To serve the Vue app
from django.conf import settings
from django.conf.urls.static import static

# Import the correct view names from your app's views.py
from executive_direction_pipeline.views import (
    UnitListView,
    MinutesOfMeetingListCreateView, 
    MinutesOfMeetingDetailView,     
    GenerateSummaryView,
    TaskAssignmentListView,         
    ExtractTasksView,
    TranscribeAndCreateMomView,
    ConvertPdfAndCreateMomView,
)

# It's good practice to group API urls under 'api/' and potentially within the app
# Create a urls_api.py file inside your executive_direction_pipeline app folder
# For now, we'll define them here, but consider moving them later.

api_urlpatterns = [
    # /api/units/ - List all units
    path('units/', UnitListView.as_view(), name='unit-list'),

    # /api/moms/ - List all MoMs or create a new one
    path('moms/', MinutesOfMeetingListCreateView.as_view(), name='mom-list-create'),

    # /api/moms/<pk>/ - Retrieve, Update, Delete a specific MoM
    path('moms/<int:pk>/', MinutesOfMeetingDetailView.as_view(), name='mom-detail'),

    # /api/moms/<pk>/generate-summary/ - Generate summary for a specific MoM
    path('moms/<int:pk>/generate-summary/', GenerateSummaryView.as_view(), name='mom-generate-summary'),

    # /api/moms/<pk>/extract-tasks/ - Extract tasks for a specific MoM
    path('moms/<int:pk>/extract-tasks/', ExtractTasksView.as_view(), name='mom-extract-tasks'),

    # /api/moms/<mom_pk>/tasks/ - List tasks related to a specific MoM
    path('moms/<int:mom_pk>/tasks/', TaskAssignmentListView.as_view(), name='mom-task-list'),

    path('moms/transcribe/', TranscribeAndCreateMomView.as_view(), name='mom-transcribe-create'),

    path('moms/convert-pdf/', ConvertPdfAndCreateMomView.as_view(), name='mom-pdf-create'),
]


urlpatterns = [
    path('admin/', admin.site.urls),

    # Include all API endpoints under the 'api/' prefix
    path('api/', include(api_urlpatterns)),

    # Serve the Vue.js frontend from the root URL
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)