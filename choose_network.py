id1='12345678'
id2='87654321'

#split ids to digits and sum them
sum1 = sum(int(digit) for digit in id1)
sum2 = sum(int(digit) for digit in id2)
total_sum = sum1 + sum2

print(f"total sum of digits in both ids is: {total_sum}")
output_digit = sum(int(digit) for digit in str(total_sum))

print(f"output digit is: {output_digit}. selected network is:")
if output_digit <= 3:
    print("ResNet18")
elif output_digit <= 6:
    print("VGG16")
elif output_digit <= 9:
    print("MobileNet V3")
else:
    print("None of the above, something went wrong")