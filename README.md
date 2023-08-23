# Video Content Restriction using Age Classification

Present-day video players have a restricted mode that can easily be toggled ”on” and young
viewers may fall prey to watching inappropriate content. The user manually signs up with
an age that can easily be falsified. Our project is a smart video player which estimates the age
of users in real-time and thus is more effective compared to the existing methods. It ensures
that a user who has not verified his age with his image is not granted access to view any of
the age-restricted content or clear the log details in the application.

A user is in safe browsing mode by default in our application, with disturbing content blocked.
He/she may play such a video on request, only after their age is verified and estimated to be
an adult. The user may request access by enabling the age-restricted content toggle button,
which initiates an image capture of the user. From the image being captured the face is
cropped and the age and gender of the user are estimated with the convolutional neural network
model supported with Gabor filters which are run on a flask server hosted on google cloud.

A log is maintained that records the details of access to age-restricted content which helps
in monitoring the use of applications by minors and to ensure that the access was denied to
such video content. This log can be cleared by an adult user only, thus preventing minors
from clearing browsing content.
