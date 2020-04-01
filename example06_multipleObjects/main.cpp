// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
	struct SampleWindow : public GLFCameraWindow
	{
		SampleWindow(const std::string& title,
			const std::vector<TriangleMesh>& model,
			const Camera& camera,
			const float worldScale)
			: GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
			sample(model)
		{
			sample.setCamera(camera);
		}

		virtual void render() override
		{
			if (cameraFrame.modified) {
				sample.setCamera(Camera{ cameraFrame.get_from(),
																 cameraFrame.get_at(),
																 cameraFrame.get_up() });
				cameraFrame.modified = false;
			}
			sample.render();
		}
		
		virtual void draw() override
		{
			sample.downloadPixels(pixels.data());
			if (fbTexture == 0)
				glGenTextures(1, &fbTexture);

			glBindTexture(GL_TEXTURE_2D, fbTexture);
			GLenum texFormat = GL_RGBA;
			GLenum texelType = GL_UNSIGNED_BYTE;
			glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
				texelType, pixels.data());

			glDisable(GL_LIGHTING);
			glColor3f(1, 1, 1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, fbTexture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glDisable(GL_DEPTH_TEST);

			glViewport(0, 0, fbSize.x, fbSize.y);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

			glBegin(GL_QUADS);
			{
				glTexCoord2f(0.f, 0.f);
				glVertex3f(0.f, 0.f, 0.f);

				glTexCoord2f(0.f, 1.f);
				glVertex3f(0.f, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 1.f);
				glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 0.f);
				glVertex3f((float)fbSize.x, 0.f, 0.f);
			}
			glEnd();
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			//// Reset transformations
			//glLoadIdentity();

			//glEnable(GL_LIGHTING);
			//glEnable(GL_LIGHT0);
			//glBegin(GL_TRIANGLES);

			//GLfloat cyan[] = { 0.f, 0.8f, 0.8f, 1.0f };
			//glMaterialfv(GL_FRONT, GL_DIFFUSE, cyan);
			//
			//for (int j = 0; j < index.size(); j++) {
			//	vec3ui idx = index[j];
			//	vec3f A = { vertex[idx.x].x, vertex[idx.x].y, vertex[idx.x].z };
			//	vec3f B = { vertex[idx.y].x, vertex[idx.y].y, vertex[idx.y].z };
			//	vec3f C = { vertex[idx.z].x, vertex[idx.z].y, vertex[idx.z].z };
			//	vec3f normal = cross(B - A, C - A);
			//	normal = normalize(normal);
			//	glNormal3f(normal.x, normal.y, normal.z);
			//	//glm::vec3 norm = normalize(A);
			//	//glNormal3f(norm.x, norm.y, norm.z);
			//	glVertex3f(A.x, A.y, A.z);
			//	//norm = normalize(B);
			//	//glNormal3f(norm.x, norm.y, norm.z);
			//	glVertex3f(B.x, B.y, B.z);
			//	//norm = normalize(C);
			//	//glNormal3f(norm.x, norm.y, norm.z);
			//	glVertex3f(C.x, C.y, C.z);
			//}


			//glEnd();
			
		}

		virtual void resize(const vec2i& newSize)
		{
			fbSize = newSize;
			sample.resize(newSize);
			pixels.resize(newSize.x * newSize.y);
		}

		vec2i                 fbSize;
		GLuint                fbTexture{ 0 };
		SampleRenderer        sample;
		std::vector<uint32_t> pixels;
		std::vector<vec3f>	  vertex;
		std::vector<vec3ui>	  index;
	};


	/*! main entry point to this example - initially optix, print hello
		world, then exit */
	extern "C" int main(int ac, char** av)
	{
		try {
			std::vector<TriangleMesh> model;
			// // 100x100 thin ground plane
			// model[0].color = vec3f(0.f, 1.f, 0.f);
			// model[0].addCube(vec3f(0.f,-1.5f, 0.f),vec3f(10.f,.1f,10.f));
			// // a unit cube centered on top of that
			// model[1].color = vec3f(0.f,1.f,1.f);
			// model[1].addCube(vec3f(0.f,1.f,0.f),vec3f(2.f,2.f,2.f));
			float red = 193.0f / 256.0f;
			float green = 215.0f / 256.0f;
			float blue = 229.0f / 256.0f;
			glClearColor(red, green, blue, 1.0);
			//making a gigantic square room
			float side_len = 50.0f;
			TriangleMesh room;
			room.color = vec3f(red, green, blue);
			float width = 0.1f;
			room.addCube(vec3f(0.f, -side_len / 2 - width / 2, 0.f), vec3f(side_len, width, side_len));
			room.addCube(vec3f(0.f, side_len / 2 + width / 2, 0.f), vec3f(side_len, width, side_len));
			room.addCube(vec3f(side_len / 2 + width / 2, 0, 0.f), vec3f(width, side_len, side_len));
			room.addCube(vec3f(-side_len / 2 - width / 2, 0, 0.f), vec3f(width, side_len, side_len));
			room.addCube(vec3f(0.f, 0, side_len / 2 + width / 2), vec3f(side_len, side_len, width));
			room.addCube(vec3f(0.f, 0, -side_len / 2 - width / 2), vec3f(side_len, side_len, width));
			model.push_back(room);

			// making a dummy "microphone"
			TriangleMesh micMesh;
			micMesh.color = vec3f(0.f, 1.f, 1.f);
			micMesh.addSphere(vec3f(1.f, 0.f, 0.f), 0.5f, 6);
			model.push_back(micMesh);

			/*TriangleMesh micMesh;
			micMesh.color = vec3f(0.f, 1.f, 1.f);
			micMesh.addSphere(vec3f(-2.f, 0.f, 0.f), 0.5f, 0);
			model.push_back(micMesh);

			TriangleMesh micMesh1;
			micMesh1.color = vec3f(0.f, 1.f, 1.f);
			micMesh1.addSphere(vec3f(-1.f, 0.f, 0.f), 0.5f, 1);
			model.push_back(micMesh1);

			TriangleMesh micMesh2;
			micMesh2.color = vec3f(0.f, 1.f, 1.f);
			micMesh2.addSphere(vec3f(0.f, 0.f, 0.f), 0.5f, 2);
			model.push_back(micMesh2);

			TriangleMesh micMesh3;
			micMesh3.color = vec3f(0.f, 1.f, 1.f);
			micMesh3.addSphere(vec3f(1.f, 0.f, 0.f), 0.5f, 3);
			model.push_back(micMesh3);

			TriangleMesh micMesh4;
			micMesh4.color = vec3f(0.f, 1.f, 1.f);
			micMesh4.addSphere(vec3f(2.f, 0.f, 0.f), 0.5f, 4);
			model.push_back(micMesh4);

			TriangleMesh micMesh5;
			micMesh5.color = vec3f(0.f, 1.f, 1.f);
			micMesh5.addSphere(vec3f(2.f, 0.f, 0.f), 0.5f, 5);
			model.push_back(micMesh5);*/
			
			//model[1].addCube(vec3f(0.f, 0.f, 0.f), vec3f(2.f, 2.f, 2.f));
			//Camera camera = { /*from*/vec3f(-4.f,-4.f,-4.f),
				Camera camera = { /*from*/vec3f(0.f,0.f,10.f),
				/* at */vec3f(0.f,0.f,0.f),
				/* up */vec3f(0.f,1.f,0.f) };
			// something approximating the scale of the world, so the
			// camera knows how much to move for any given user interaction:
			const float worldScale = 10.f;

			//SampleWindow* window = new SampleWindow("Optix 7 Course Example", model, camera, worldScale);
			SampleRenderer* renderer = new SampleRenderer(model);
			SoundSource* src = new SoundSource();
			Microphone* mic = new Microphone();
			if (ac > 1) {
				src->num_rays = atoi(av[1]);
			}
			renderer->add_mic(mic);
			renderer->add_source(src);
			renderer->auralize();
			//window->run();

		}
		catch (std::runtime_error & e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
				<< GDT_TERMINAL_DEFAULT << std::endl;
			exit(1);
		}
		return 0;
	}

} // ::osc
