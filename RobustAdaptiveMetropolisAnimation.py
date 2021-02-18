from manim import *
import numpy as np

class RAM(Scene):

	def target_ellipse(self, t):
		# Target Cholesky Factor
		L_pi = np.sqrt(-np.log(0.05) / 50) * np.array([[np.sqrt(101), 0, 0],
										[99 / np.sqrt(101), np.sqrt(101 - ((99 ** 2) / 101)), 0],
										[0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(L_pi.dot(unit_circle))

	def ellipse_1(self, t):
		S_1 = np.sqrt(-2 * np.log(0.05)) * np.array([[1, 0, 0],
					    							 [0, 1, 0],
					    							 [0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(S_1.dot(unit_circle))

	def ellipse_2(self, t):
		S_2 = np.sqrt(-2 * np.log(0.05)) * np.array([[0.88189621, 0, 0],
													 [0.05792464, 0.9924232, 0],
													 [0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(S_2.dot(unit_circle))

	def ellipse_3(self, t):
		S_3 = np.sqrt(-2 * np.log(0.05)) * np.array([[0.8694355, 0, 0],
													 [0.1145327, 0.9313289, 0],
													 [0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(S_3.dot(unit_circle))

	def ellipse_4(self, t):
		S_4 = np.sqrt(-2 * np.log(0.05)) * np.array([[0.8485996, 0, 0],
													 [0.1631072, 0.9009765, 0],
													 [0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(S_4.dot(unit_circle))

	def ellipse_5(self, t):
		S_5 = np.sqrt(-2 * np.log(0.05)) * np.array([[0.8212996, 0, 0],
													 [0.1951626, 0.8886948, 0],
													 [0, 0, 1]])
		unit_circle = np.array([np.cos(t), np.sin(t), 0])
		return(S_5.dot(unit_circle))

	def step(self, Y, X, X_label_old, X_coord, alpha, accepted, n, Sigma, Sigma_old, S, old_ellipse, wait_time = 0.3):
		"""Move the animation along one step in the MC.
		Y: (array) Proposal Coords
		X: (array) Current state in the chain
		X_label_old: (MathTex) label attached to the current state
		X_coord: (SmallDot) The point at the current state
		alpha: (float) Acceptance probability
		accepted: (bool) Whether the proposed point is accepted
		n: (int) Which iteration we are in
		Sigma: (array) The updated covariance
		Sigma_old: (MathTex) The LaTeX of the current sigma
		S: (array) The Cholesky factor of the new covariance
		old_ellipse: (function) The 0.95 contour of the old proposal
		wait_time: (float) Time (in seconds) to wait between animations"""

		Y_coord = SmallDot(radius = 0.09, color = RED).shift([Y[0], Y[1], 0])
		Y_label_text = r"Y_" + str(n)
		Y_label = MathTex(Y_label_text).scale(0.6).shift([Y[0] + 0.3, Y[1], 0])
		alpha_2dp = '{0:.2f}'.format(alpha)
		alpha_value_text = r"\alpha_" + str(n) + r" = " + alpha_2dp
		alpha_value = MathTex(alpha_value_text).scale(0.6).shift([5, -0.5, 0])

		self.play(
			FadeIn(Y_coord),
			Write(Y_label)
		)
		self.wait(wait_time)
		self.play(FadeIn(alpha_value))
		self.wait(wait_time)

		if accepted:
			acceptance = MathTex(r"\checkmark", color = GREEN).shift([6.1, -0.5, 0])
			self.play(FadeInFrom(acceptance, LEFT))
			self.wait(wait_time)

			def new_ellipse(t): # changed self, t to t
				S_ellipse = np.sqrt(-2 * np.log(0.05)) * np.array([[S[0][0], 0, 0],
														   		   [S[1][0], S[1][1], 0],
														   		   [0, 0, 1]])
				unit_circle = np.array([np.cos(t), np.sin(t), 0])
				return(np.array([Y[0], Y[1], 0]) + S_ellipse.dot(unit_circle))

			Sigma_new_inner = '{0:.2f}'.format(Sigma[0][0]) + r" & " + '{0:.2f}'.format(Sigma[0][1]) + r"\\" + '{0:.2f}'.format(Sigma[1][0]) + r" & " + '{0:.2f}'.format(Sigma[1][1])
			Sigma_new_text = r"S_" + str(n) + r"S_" + str(n) + r"^T = \begin{pmatrix}" + Sigma_new_inner + r"\end{pmatrix}"
			Sigma = MathTex(Sigma_new_text).scale(0.6).to_corner([1, 1, 0])
			e = ParametricFunction(new_ellipse, t_min = 0, t_max = 2 * PI, color = RED) # changed from self.new_ellipse, to new_ellipse
			X_label_text = r"X_" + str(n)
			X_label = MathTex(X_label_text).scale(0.6).shift([Y[0] + 0.25, Y[1] - 0.25, 0])

			self.play(
				ReplacementTransform(Sigma_old, Sigma),
				ReplacementTransform(old_ellipse, e)
			)
			self.wait(wait_time)
			self.play(
				FadeOut(Y_label),
				FadeOut(alpha_value),
				FadeOut(acceptance),
				ReplacementTransform(X_coord, Y_coord),
				ReplacementTransform(X_label_old, X_label),
				Y_coord.animate.set_fill(WHITE)
			)
			self.wait(wait_time)

			return({'X': Y,
					'X_label': X_label,
					'X_coord': Y_coord,
					'Sigma': Sigma,
					'ellipse': e})
		else:
			rejection = MathTex(r"\times", color = RED).shift([6.1, -0.5, 0])
			self.play(FadeInFrom(rejection, LEFT))
			self.wait(wait_time)

			def new_ellipse(t): # changed self, t to t
				S_ellipse = np.sqrt(-2 * np.log(0.05)) * np.array([[S[0][0], 0, 0],
														   		   [S[1][0], S[1][1], 0],
														   		   [0, 0, 1]])
				unit_circle = np.array([np.cos(t), np.sin(t), 0])
				return(np.array([X[0], X[1], 0]) + S_ellipse.dot(unit_circle))

			Sigma_new_inner = '{0:.2f}'.format(Sigma[0][0]) + r" & " + '{0:.2f}'.format(Sigma[0][1]) + r"\\" + '{0:.2f}'.format(Sigma[1][0]) + r" & " + '{0:.2f}'.format(Sigma[1][1])
			Sigma_new_text = r"S_" + str(n) + r"S_" + str(n) + r"^T = \begin{pmatrix}" + Sigma_new_inner + r"\end{pmatrix}"
			Sigma = MathTex(Sigma_new_text).scale(0.6).to_corner([1, 1, 0])
			e = ParametricFunction(new_ellipse, t_min = 0, t_max = 2 * PI, color = RED) # changed from self.new_ellipse, to new_ellipse
			X_label_text = r"X_" + str(n)
			X_label = MathTex(X_label_text).scale(0.6).shift([X[0] + 0.25, X[1] - 0.25, 0])

			self.play(
				ReplacementTransform(Sigma_old, Sigma),
				ReplacementTransform(old_ellipse, e)
			)
			self.wait(wait_time)
			self.play(
				FadeOut(Y_coord),
				FadeOut(Y_label),
				FadeOut(alpha_value),
				FadeOut(rejection),
				ReplacementTransform(X_label_old, X_label)
			)
			self.wait(wait_time)

			return({'X': X,
					'X_label': X_label,
					'X_coord': X_coord,
					'Sigma': Sigma,
					'ellipse': e})


	def construct(self):
		# Set up the scene
		title = Tex("Robust Adaptive Metropolis")
		transform_title = Tex("RAM").scale(0.6)
		self.play(Write(title))
		self.wait()
		transform_title.to_corner(UP + LEFT)
		self.play(ReplacementTransform(title, transform_title))
		grid = NumberPlane(background_line_style = {"stroke_color": BLUE_D,
            										"stroke_width": 2,
            										"stroke_opacity": 0.25})
		self.add(grid)
		self.bring_to_front(transform_title)
		self.play(ShowCreation(grid, run_time = 3, lag_ratio = 0.1))
		self.wait(1.5)
		target_covar = MathTex(r"\Sigma_{\pi} = \frac{1}{100} \begin{pmatrix}101 & 99\\99 & 101\end{pmatrix}").scale(0.6)
		target_covar.to_corner(UP + LEFT)
		target_covar.shift(DOWN * 0.35)
		self.play(FadeInFrom(target_covar, direction = UP))
		
		targ_ellipse = ParametricFunction(self.target_ellipse,
										  t_min = 0,
										  t_max = 2 * PI,
										  color = YELLOW, 
										  fill_opacity = 0)
		target_chol = MathTex(r"L_{\pi} = \frac{1}{10} \begin{pmatrix}\sqrt{101} & 0\\ \frac{99}{\sqrt{101}} & \sqrt{101 - \frac{99^2}{101}}\end{pmatrix}").scale(0.6)
		target_chol.to_corner(UP + LEFT)
		target_chol.shift(DOWN * 1.1)
		self.play(DrawBorderThenFill(targ_ellipse))
		self.play(FadeInFrom(target_chol, direction = UP))
		self.wait(1.5)

		# now animate the MCMC
		x_1 = SmallDot(radius = 0.09)
		x_1_label = MathTex(r"x_1").scale(0.6).shift([0.25, -0.25, 0])
		self.play(
			FadeIn(x_1),
			Write(x_1_label),
			targ_ellipse.animate.set_stroke(opacity = 0.25)
		)
		self.wait(1.5)

		# choose from the N(0, I) distn.
		U_distn = MathTex(r"U_2 \sim N(0, I)").scale(0.6).shift([5.8, -0.4, 0])
		U_2_value = MathTex(r"U_2= \begin{pmatrix}-1.21\\ 0.28\end{pmatrix}").scale(0.6).shift([5.81, -1, 0])
		e_1 = ParametricFunction(self.ellipse_1,
								 t_min = 0,
								 t_max = 2 * PI,
								 color = RED,
								 fill_opacity = 0)
		U_2 = SmallDot(radius = 0.09, color = RED).shift([-1.2070657, 0.2774292, 0])
		U_2_label = MathTex(r"U_2").scale(0.6).shift([-1.2070657 + 0.3, 0.2774292, 0])
		self.play(FadeIn(U_distn))
		self.wait(1.5)
		self.play(DrawBorderThenFill(e_1))
		self.wait(1.5)
		self.play(FadeInFrom(U_2_value, direction = UP))
		self.wait(1.5)
		self.play(
			FadeIn(U_2),
			FadeIn(U_2_label)
		)
		self.wait(1.5)

		Y_2_equation = MathTex(r"Y_2 = x_1 + S_1 U_2 = U_2").scale(0.6).shift([5.4, -1.15, 0])
		# Y_1_value = MathTex(r"Y_1 = \begin{pmatrix}0.755\\ 0.655\end{pmatrix}").scale(0.6).shift([6, -1.5, 0])
		Y_2_label = MathTex(r"Y_2").scale(0.6).shift([-1.2070657 + 0.3, 0.2774292, 0])

		self.play(
			FadeOut(U_distn),
			U_2_value.animate.move_to([5.8, -0.5, 0]),
			FadeIn(Y_2_equation)
		)
		self.wait(1.5)
		self.play(ReplacementTransform(U_2_label, Y_2_label))

		alpha_2_value = MathTex(r"\alpha_2 = 10^{-12}").scale(0.6).shift([5, -1.6, 0])
		rejection = MathTex(r"\times", color = RED).shift([6, -1.6, 0])
		X_2_label = MathTex(r"X_2").scale(0.6).shift([0.25, -0.25, 0])
		self.play(FadeIn(alpha_2_value))
		self.wait(1)
		self.play(Write(rejection))
		self.wait(1.5)
		self.play(
			FadeOut(Y_2_label),
			FadeOut(U_2)
		)
		self.wait()
		self.play(ReplacementTransform(x_1_label, X_2_label))
		self.wait(1.5)

		
		adaptation = MathTex(r"S_2S_2^T = S_1\left(I+\eta_1(\alpha_1 - \alpha_*)\frac{U_2 U_2^T}{|U_2|^2}\right)S_1^T")
		adaptation_value = MathTex(r"= \begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}+\frac{1}{1^{0.7}}(10^{-12} - 0.234)\frac{1}{1.53}\begin{pmatrix}-1.21\\0.28\end{pmatrix}\begin{pmatrix}-1.21 & 0.28\end{pmatrix}")
		adaptation_final = MathTex(r"= \begin{pmatrix}0.77 & 0.05\\0.05 & 0.99\end{pmatrix}")

		adaptation.scale(0.4).to_corner([1, 1, 0]).shift([-2, 0.3, 0])
		adaptation_value.scale(0.4).move_to(adaptation.get_center() + [1.5, -0.7, 0])
		adaptation_final.scale(0.4).move_to(adaptation.get_center() + [-0.49, -1.4, 0])

		Sigma_2 = MathTex(r"S_2S_2^T = \begin{pmatrix}0.77 & 0.05\\0.05 & 0.99\end{pmatrix}").scale(0.6).to_corner([1, 1, 0])

		e_2 = ParametricFunction(self.ellipse_2,
								 t_min = 0,
								 t_max = 2 * PI,
								 color = RED)

		self.play(FadeIn(adaptation))
		self.wait(1.5)
		self.play(FadeIn(adaptation_value))
		self.wait(1.5)
		self.play(FadeIn(adaptation_final))
		self.wait(1.5)
		self.play(
			ReplacementTransform(e_1, e_2),
			FadeOut(U_2_value),
			FadeOut(Y_2_equation),
			FadeOut(alpha_2_value),
			FadeOut(rejection),
			FadeOut(adaptation),
			FadeOut(adaptation_value),
			FadeOut(adaptation_final),
			FadeIn(Sigma_2)
		)
		self.wait(1.5)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Y_3 = SmallDot(radius = 0.09, color = RED).shift([0.3168557, -0.7041268, 0])
		Y_3_label = MathTex(r"Y_3").scale(0.6).shift([0.3168557 + 0.3, -0.7041268, 0])
		alpha_3_value = MathTex(r"\alpha_3 = 2 \times 10^{-6}").scale(0.6).shift([5, -0.5, 0])
		rejection.move_to([6.1, -0.5, 0])

		self.play(
			FadeIn(Y_3),
			Write(Y_3_label)
		)
		self.wait(1.5)
		self.play(FadeIn(alpha_3_value))
		self.wait()
		self.play(FadeInFrom(rejection, LEFT))
		self.wait(1.5)

		Sigma_3 = MathTex(r"S_3S_3^T = \begin{pmatrix}0.76 & 0.10\\0.10 & 0.88\end{pmatrix}").scale(0.6).to_corner([1, 1, 0])
		e_3 = ParametricFunction(self.ellipse_3, t_min = 0, t_max = 2 * PI, color = RED)
		X_3_label = MathTex(r"X_3").scale(0.6).shift([0.25, -0.25, 0])

		self.play(
			ReplacementTransform(Sigma_2, Sigma_3),
			ReplacementTransform(e_2, e_3)
		)
		self.wait()
		self.play(
			FadeOut(Y_3),
			FadeOut(Y_3_label),
			FadeOut(alpha_3_value),
			FadeOut(rejection),
			ReplacementTransform(X_2_label, X_3_label)
		)
		self.wait(1.5)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Y_4 = SmallDot(radius = 0.09, color = RED).shift([0.4399830, -0.4773120, 0])
		Y_4_label = MathTex(r"Y_4").scale(0.6).shift([0.4399830 + 0.3, -0.4773120, 0])
		alpha_4_value = MathTex(r"\alpha_4 = 3 \times 10^{-5}").scale(0.6).shift([5, -0.5, 0])

		self.play(
			FadeIn(Y_4),
			Write(Y_4_label)
		)
		self.wait(1.5)
		self.play(FadeIn(alpha_4_value))
		self.wait()
		self.play(FadeInFrom(rejection, LEFT))
		self.wait(0.3)

		Sigma_4 = MathTex(r"S_4S_4^T = \begin{pmatrix}0.72 & 0.14\\0.14 & 0.84\end{pmatrix}").scale(0.6).to_corner([1, 1, 0])
		e_4 = ParametricFunction(self.ellipse_4, t_min = 0, t_max = 2 * PI, color = RED)
		X_4_label = MathTex(r"X_4").scale(0.6).shift([0.25, -0.25, 0])

		self.play(
			ReplacementTransform(Sigma_3, Sigma_4),
			ReplacementTransform(e_3, e_4)
		)
		self.wait(0.3)
		self.play(
			FadeOut(Y_4),
			FadeOut(Y_4_label),
			FadeOut(alpha_4_value),
			FadeOut(rejection),
			ReplacementTransform(X_3_label, X_4_label)
		)
		self.wait(0.3)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		Y_5 = SmallDot(radius = 0.09, color = RED).shift([0.8345161, -0.4004187, 0])
		Y_5_label = MathTex(r"Y_5").scale(0.6).shift([0.8345161 + 0.3, -0.4004187, 0])
		alpha_5_value = MathTex(r"\alpha_4 = 5 \times 10^{-9}").scale(0.6).shift([5, -0.5, 0])

		self.play(
			FadeIn(Y_5),
			Write(Y_5_label)
		)
		self.wait(0.3)
		self.play(FadeIn(alpha_5_value))
		self.wait(0.3)
		self.play(FadeInFrom(rejection, LEFT))
		self.wait(0.3)

		Sigma_5 = MathTex(r"S_5S_5^T = \begin{pmatrix}0.67 & 0.16\\0.16 & 0.83\end{pmatrix}").scale(0.6).to_corner([1, 1, 0])
		e_5 = ParametricFunction(self.ellipse_5, t_min = 0, t_max = 2 * PI, color = RED)
		X_5_label = MathTex(r"X_5").scale(0.6).shift([0.25, -0.25, 0])

		self.play(
			ReplacementTransform(Sigma_4, Sigma_5),
			ReplacementTransform(e_4, e_5)
		)
		self.wait(0.3)
		self.play(
			FadeOut(Y_5),
			FadeOut(Y_5_label),
			FadeOut(alpha_5_value),
			FadeOut(rejection),
			ReplacementTransform(X_4_label, X_5_label)
		)
		self.wait(0.3)


		step_6 = self.step(Y = np.array([-0.3919182, -0.9803910]),
				  		   X = np.array([0, 0]),
				  		   X_label_old = X_5_label,
				  		   X_coord = x_1,
				  		   alpha = 0.01,
				  		   accepted = False,
				  		   n = 6,
				  		   Sigma = np.array([[0.6654424, 0.1375466], [0.1375466, 0.7709815]]),
				  		   Sigma_old = Sigma_5,
				  		   S = np.array([[0.8157465, 0], [0.1686144, 0.8617138]]),
				  		   old_ellipse = e_5)

		step_7 = self.step(Y = np.array([0.7179419, 1.3289551]),
				  		   X = step_6['X'],
				  		   X_label_old = step_6['X_label'],
				  		   X_coord = step_6['X_coord'],
				  		   alpha = 0.01,
				  		   accepted = False,
				  		   n = 7,
				  		   Sigma = np.array([[0.6527737, 0.114096], [0.114096, 0.727573]]),
				  		   Sigma_old = step_6['Sigma'],
				  		   S = np.array([[0.8079441, 0], [0.1412177, 0.8412078]]),
				  		   old_ellipse = step_6['ellipse'])

		step_8 = self.step(Y = np.array([-0.08910452, -0.44543945]),
				  		   X = step_7['X'],
				  		   X_label_old = step_7['X_label'],
				  		   X_coord = step_7['X_coord'],
				  		   alpha = 0.20,
				  		   accepted = True,
				  		   n = 8,
				  		   Sigma = np.array([[0.6525008, 0.1127319], [0.1127319, 0.7207537]]),
				  		   Sigma_old = step_7['Sigma'],
				  		   S = np.array([[0.8077752, 0], [0.1395585, 0.8374229]]),
				  		   old_ellipse = step_7['ellipse'])








		

		





























