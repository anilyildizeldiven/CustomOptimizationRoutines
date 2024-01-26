import tensorflow as tf
import matplotlib.pyplot as plt


# Quadratische Testfunktion
def quadratic_function(x):
    return tf.reduce_sum(tf.square(x - 5))

# Initialisierung
x_init_value = [10.0, 10.0]
num_steps = 100

# Loss-Verlauf speichern
loss_progress_custom = []
loss_progress_gd = []

# Optimierer
optimizer_custom = TestGD()
optimizer_gd = tf.optimizers.SGD(learning_rate=0.1)

# Trainiere mit benutzerdefiniertem Optimierer
x_custom = tf.Variable(x_init_value)
for step in range(num_steps):
    with tf.GradientTape() as tape:
        loss = quadratic_function(x_custom)
    grads = tape.gradient(loss, [x_custom])
    optimizer_custom.apply_gradients(zip(grads, [x_custom]))
    loss_progress_custom.append(loss.numpy())

# Trainiere mit Standard-GD
x_gd = tf.Variable(x_init_value)
for step in range(num_steps):
    with tf.GradientTape() as tape:
        loss = quadratic_function(x_gd)
    grads = tape.gradient(loss, [x_gd])
    optimizer_gd.apply_gradients(zip(grads, [x_gd]))
    loss_progress_gd.append(loss.numpy())

# Ergebnisse visualisieren
plt.figure(figsize=(10, 6))
plt.plot(loss_progress_custom, label='Custom Optimizer')
plt.plot(loss_progress_gd, label='Gradient Descent')
plt.xlabel('Iterationen')
plt.ylabel('Loss')
plt.title('Vergleich der Konvergenzgeschwindigkeit')
plt.legend()
plt.show()
