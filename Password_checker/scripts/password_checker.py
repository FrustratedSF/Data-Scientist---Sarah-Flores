# This Python program checks the strength of a password based on its length whether the password written is weak, medium or strong.


while True:
  password = input('Enter your password: ')
  length = len(password)

if length < 5:
  print('Weak password! Try again')

elif length < 10:
  print('Medium password! Try again')

else:
  print('Strong password')
  break
