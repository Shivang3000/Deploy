from django.contrib.auth.models import User
from rest_framework import serializers

class UserSerializer (serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        min_length=3,
        style={'input_type':'password'}
        )

    class Meta:
        model = User
        fields = ['username','email','password']

    def create(self,validator_data):
        user = User.objects.create_user(
            validator_data['username'],
            validator_data['email'],
            validator_data['password']
        )

        return user