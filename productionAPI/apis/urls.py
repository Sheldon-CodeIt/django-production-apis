from django.urls import path
from . import views


urlpatterns = [
    path('', views.Home),
    path('predict-mental-health/', views.PredictMentalHealth, name="predict-mental-health"),
]

