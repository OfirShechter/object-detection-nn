id1='12345678'
id2='87654321'

#split ids to digits and sum them
sum1 = sum(int(digit) for digit in id1)
sum2 = sum(int(digit) for digit in id2)
digit_sum = str(sum1 + sum2)
print(f"total sum of digits is: {digit_sum}")

# repeated digit sum
while len(digit_sum) > 1:
    digit_sum = str(sum(int(digit) for digit in digit_sum))
    print(f"total sum of digits is: {digit_sum}")

output_digit = int(digit_sum)
print(f"output digit is: {output_digit}. selected network is:")
if output_digit <= 3:
    print("ResNet18")
elif output_digit <= 6:
    print("VGG16")
elif output_digit <= 9:
    print("MobileNet V3")
else:
    print("None of the above, something went wrong")
    
# total sum of digits is: 78
# total sum of digits is: 15
# total sum of digits is: 6
# output digit is: 6. selected network is:
# VGG16