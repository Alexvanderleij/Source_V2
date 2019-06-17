#undef UNICODE
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <math.h>
#include <ctime>
#include <direct.h>
#include <curl/curl.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#pragma comment (lib, "Ws2_32.lib")

#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "27015"

using namespace std;
using namespace cv;
using namespace cv::ocl;
using namespace cv::dnn;

string namesFile = "coco.names";
string cfgFile = "yolov3.cfg";
string weightsFile = "yolov3.weights";
const char* axisCameraURL = "http://192.168.253.205/axis-cgi/jpg/image.cgi?camera=quad";

double timeBetweenSnapshots = 0.100; //[ms] minimal time, only used when processing an image requires less time than this
double timeNoBrokenBeamsEndOfEvent = 0.750; //[ms]

//left to right = in // Right to left = out
bool leftTrigger = false, rightTrigger = false;
bool leftEvent = false, rightEvent = false;
bool takeSnaps = false;
bool processLastEvent = false;
int personsInRoom = 0;
int personsInRoomV2 = 0;
int eventCounter = 0;

//Define object detection variables
float confThreshold = 0.40;
float nmsThreshold = 0.5;
int inpWidth = 320;	//width of network's input image
int inpHeight = 320;	//height of network's input image //320 is fast, 416, 608, 832 is accurate
vector<string> names; //vector containing names of the output layers
vector<string> classes; //vector containing all detectable classes
Net net;
typedef vector<Mat> snapshots;
vector<snapshots> events;
vector<bool> inOutEvents;


int receiveTCPpackages();
void processSnapshots();
void takeSnapshots();

int main(int argc, char** argv)
{
	// Start reading TCP messages
	thread t1(receiveTCPpackages);

	//Load the names of all classes
	ifstream ifs(namesFile.c_str());
	string line;
	while (getline(ifs, line)) { classes.push_back(line); }

	//Load the neural network
	net = readNetFromDarknet(cfgFile, weightsFile);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(DNN_TARGET_OPENCL);	//GPU
	net.setPreferableTarget(DNN_TARGET_CPU);		//CPU

	//Get the names of the output layers
	if (names.empty()) {
		vector<int> outLayers = net.getUnconnectedOutLayers();
		vector<string> layerNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); i++) {
			names[i] = layerNames[outLayers[i] - 1];
		}
	}

	//Start processing all taken snapshots
	thread t2(processSnapshots);

	//Get initial number of persons inside
	personsInRoom = 0;

NextEvent:
	//Wait for event
	while (true) {
		if (leftTrigger == true) {
			leftEvent = true;
			break;
		}
		else if (rightTrigger == true) {
			rightEvent = true;
			break;
		}
		else {
			Sleep(1);
		}
	}
	takeSnaps = true;
	processLastEvent = false;
	vector<Mat> newEvent;
	events.push_back(newEvent);
	cout << "New Event!\t(total events in queue =" << events.size() << ")" << endl;

	//Take Snapshots during event
	thread t3(takeSnapshots);
	bool bothBeamTriggered = false;
	int lastTriggeredBeam = 0; //1=left 2=right 3=both
	auto lastTrigger = chrono::system_clock::now();
	while (takeSnaps == true) {

		if (leftTrigger == true || rightTrigger == true) {
			lastTrigger = chrono::system_clock::now();
		}
		auto currentTime = chrono::system_clock::now();
		chrono::duration<double> eventOverCounter = currentTime - lastTrigger;

		if (eventOverCounter.count() >= timeNoBrokenBeamsEndOfEvent) {
			takeSnaps = false;
			cout << "Event ended!" << endl;
			break;
		}

		if ((leftEvent == true && rightTrigger == true) || (rightEvent == true && leftTrigger == true)) {
			bothBeamTriggered = true;
		}

		if (leftTrigger == true && rightTrigger == false) { lastTriggeredBeam = 1; }
		else if (leftTrigger == false && rightTrigger == true) { lastTriggeredBeam = 2; }
		else if (leftTrigger == true && rightTrigger == true) { lastTriggeredBeam = 3; }
	}

	//Wait for last taken snapshot
	t3.join();

	//Check validity of event
	if (bothBeamTriggered == false) {
		cout << "False event: Only one beam triggered." << endl;
		events.erase(events.end()-1);
	}
	else if (leftEvent == true && lastTriggeredBeam == 2) {
		//leftEvent
		inOutEvents.push_back(true);
	}
	else if (rightEvent == true && lastTriggeredBeam == 1) {
		//rightEvent
		inOutEvents.push_back(false);
	}
	else {
		cout << "False event: first and last triggered beam were the same." << endl;
		events.erase(events.end()-1);
	}
	///cout << "Events size: " << events.size() << endl;

	//End of event
	processLastEvent = true;
	leftEvent = false;
	rightEvent = false;

	//Wait for next event
	goto NextEvent;


	return 0;
}


//////////////////
//  BEAMS PART  //
//////////////////

int receiveTCPpackages() {
	WSADATA wsaData;
	int iResult;

	SOCKET ListenSocket = INVALID_SOCKET;
	SOCKET ClientSocket = INVALID_SOCKET;

	struct addrinfo* result = NULL;
	struct addrinfo hints;

	int iSendResult;
	char recvbuf[DEFAULT_BUFLEN];
	int recvbuflen = DEFAULT_BUFLEN;

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed with error: %d\n", iResult);
		waitKey(0);
		return 1;
	}

	ZeroMemory(&hints, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;


	while (true) {

		// Resolve the server address and port
		iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
		if (iResult != 0) {
			printf("getaddrinfo failed with error: %d\n", iResult);
			WSACleanup();
			waitKey(0);
			return 1;
		}

		// Create a SOCKET for connecting to server
		ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
		if (ListenSocket == INVALID_SOCKET) {
			printf("socket failed with error: %ld\n", WSAGetLastError());
			freeaddrinfo(result);
			WSACleanup();
			waitKey(0);
			return 1;
		}

		// Setup the TCP listening socket
		iResult = ::bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
		if (iResult == SOCKET_ERROR) {
			printf("bind failed with error: %d\n", WSAGetLastError());
			freeaddrinfo(result);
			closesocket(ListenSocket);
			WSACleanup();
			waitKey(0);
			return 1;
		}

		freeaddrinfo(result);

		iResult = listen(ListenSocket, SOMAXCONN);
		if (iResult == SOCKET_ERROR) {
			printf("listen failed with error: %d\n", WSAGetLastError());
			closesocket(ListenSocket);
			WSACleanup();
			waitKey(0);
			return 1;
		}

		// Accept a client socket
		ClientSocket = accept(ListenSocket, NULL, NULL);
		if (ClientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", WSAGetLastError());
			closesocket(ListenSocket);
			WSACleanup();
			waitKey(0);
			return 1;
		}
		// No longer need server socket
		closesocket(ListenSocket);

		// Receive until the peer shuts down the connection
		do {

			iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
			if (iResult > 0) {

				string mess(recvbuf);
				std::cout << "received message '" << mess << "'" << endl;	//print received TCP packages
				if (mess == "L1\n" || mess == "l1") {
					leftTrigger = true;
				}
				else if (mess == "L0\n" || mess == "l0") {
					leftTrigger = false;
				}
				else if (mess == "R1\n" || mess == "r1") {
					rightTrigger = true;
				}
				else if (mess == "R0\n" || mess == "r0") {
					rightTrigger = false;
				}
				//else { std::cout << "Unknown message received from Axis: " <<  mess; }

				// Echo the buffer back to the sender
				iSendResult = send(ClientSocket, recvbuf, iResult, 0);

				//clear buffer
				memset(recvbuf, 0, sizeof recvbuf);

				if (iSendResult == SOCKET_ERROR) {
					printf("send failed with error: %d\n", WSAGetLastError());
					waitKey(0);
					closesocket(ClientSocket);
					WSACleanup();
					return 1;
				}
			}
		} while (iResult > 0);

	}



	// shutdown the connection since we're done
	iResult = shutdown(ClientSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		printf("shutdown failed with error: %d\n", WSAGetLastError());
		closesocket(ClientSocket);
		WSACleanup();
		waitKey(0);
		return 1;
	}

	// cleanup
	closesocket(ClientSocket);
	WSACleanup();
}


//////////////////
// CAMERA PART  //
//////////////////

size_t write_data(char* ptr, size_t size, size_t nmemb, void* userdata)
{
	vector<uchar>* stream = (vector<uchar>*)userdata;
	size_t count = size * nmemb;
	stream->insert(stream->end(), ptr, ptr + count);
	return count;
}





//function to retrieve the image as cv::Mat data type
Mat curlImg(const char* imageURL, int timeout = 10)
{
	vector<uchar> stream;
	CURL* curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_URL, imageURL); //the img url
	curl_easy_setopt(curl, CURLOPT_HTTPAUTH, (long)CURLAUTH_ANY);
	curl_easy_setopt(curl, CURLOPT_USERPWD, "root:admin");
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs, 
	/*CURLcode res =*/ curl_easy_perform(curl); // start curl
	curl_easy_cleanup(curl); // cleanup
	return imdecode(stream, -1); // 'keep-as-is'
}

void takeSnapshots() {
	cout << "Start taking snapshots.." << endl;

	while (takeSnaps) {
		auto lastSnap = chrono::system_clock::now();

		//Take snapshot
		Mat snapshot;
		auto imgRequestStart = chrono::system_clock::now();
		curlImg(axisCameraURL).copyTo(snapshot);

		auto imgRequestEnd = chrono::system_clock::now();
		chrono::duration<double> imgRequestTime = imgRequestEnd - imgRequestStart;
		cout << "imgRequestTime = " << imgRequestTime.count() << endl;
		if (snapshot.empty()) {
			cerr << "Snapshot couldn't be loaded." << endl;
			return;
		}

		//Cut region of interests in snapshot camera
		int PxWidth = snapshot.cols;
		int PxHeight = snapshot.rows;
		double viewWidth = 0.75; //middle 40%
		unsigned topVideo = 28, bottomVideo = PxHeight / 2 - 3;
		Mat snapLeft = snapshot(Rect(PxWidth * (1 - viewWidth) / 2 / 2 + 0, topVideo, PxWidth * viewWidth / 2, bottomVideo - topVideo));
		Mat snapRight = snapshot(Rect(PxWidth * (1 - viewWidth) / 2 / 2 + PxWidth / 2, topVideo, PxWidth * viewWidth / 2, bottomVideo - topVideo));
		Mat snapRightFlipped;
		flip(snapRight, snapRightFlipped, 1);
		Mat blackBar(80, snapLeft.cols, CV_8UC3, Scalar(0, 0, 0));
		vconcat(snapLeft, blackBar, snapshot);
		vconcat(snapshot, snapRightFlipped, snapshot);
		
		events.back().push_back(snapshot);


		//Take one snapshot every X ms
		while (true) {
			auto cur = chrono::system_clock::now();
			chrono::duration<double> timeSinceLastSnap = cur - lastSnap;
			if (timeSinceLastSnap.count() >= timeBetweenSnapshots) {
				break;
			}
			else if (takeSnaps == false) {
				break;
			}
		}
	}
	return;
}


void processSnapshots()
{
	int personsInPoort = 0, pervPersonsInPoort = 0;
	int personsPassed = 0;

	int personsPassedV2 = 0;

	processNextEvent:;

	//Wait for a finished event
	while (events.size() <= 1) {
		if (events.size() == 1 && processLastEvent == true) {
			break;
		}
		Sleep(1);
	}
	personsPassed = 0;

	vector<int>knownXpos;
	vector<int>countXpos;
	personsPassedV2 = 0;

	//Process oldest event
	cout << "Process event..." << endl;
	
	//create 4D blob and set it as input
	Mat blob;
	blobFromImages(events.at(0), blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	vector<Mat> outs;

	
	//feed forward the blob through the network
	auto startProc = chrono::system_clock::now();
	net.forward(outs, names);
	auto endProc = chrono::system_clock::now();
	chrono::duration<double> procTime = endProc - startProc;
	cout << "Processing Time (batch size = " << events[0].size() << " snapshots): " << fixed << setprecision(3) << procTime.count() << "s. Average time per snapshot: " << procTime.count() / events[0].size() << "s (" << events[0].size() / procTime.count() << " fps)." << endl;
	///cout << "Show result.." << endl;
	///cout << "Persons in gate: ";
	
	int snapCounter = 0;
	 
	//process output
	for (int image = 0; image < outs[0].size[0]; image++) {			//outs[x].size[0] is equal to the number of input images
		vector<Rect> bboxes;
		vector<float> confidences;
		vector<int> classIds;
		for (int j = 0; j < outs.size(); j++) {			//outs.size is equal to the number of output layers (3 for yolov3)
			for (int k = 0; k < outs[j].size[1]; k++) {		//read every output layer row by row
				float confidence = 0.00;
				int classID = -1;
				///int ll = 0;
				for (int l = 5; l < outs[j].size[2]; l++) {
				if (outs[j].at<float>(image, k, l) >= confidence) {
						confidence = outs[j].at<float>(image, k, l);
						classID = l - 5;
						///ll = l;
					}
				}

				if (confidence >= confThreshold && classID == 0) {
					///cout << "(j){i,k,l} = (" << j << ") {" << image << "," << k << "," << ll << "}" << endl;
					float x = outs[j].at<float>(image, k, 0);
					float y = outs[j].at<float>(image, k, 1);
					float width = outs[j].at<float>(image, k, 2);
					float height = outs[j].at<float>(image, k, 3);

					int xLeftBottom = static_cast<int>((x - width / 2) * events.at(0).at(image).cols);
					int yLeftBottom = static_cast<int>((y - height / 2) * events.at(0).at(image).rows);
					int xRightTop = static_cast<int>((x + width / 2) * events.at(0).at(image).cols);
					int yRightTop = static_cast<int>((y + height / 2) * events.at(0).at(image).rows);
					Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
					bboxes.push_back(object);
					confidences.push_back(confidence);
					classIds.push_back(classID);
				}
			}
		}

		//Delete all persons outside the gate
		Mat ROIzone = Mat::zeros(Size(events.at(0).at(image).cols, events.at(0).at(image).rows), CV_8UC3);
		vector<Point> ROIpoly;
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 24 / 100, 0));	//top left topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 83 / 100, 0));	//top right topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 63 / 100, events.at(0).at(image).rows * 38 / 100));	//bottom right topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 63 / 100, events.at(0).at(image).rows * 46 / 100));	//ground right topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 78 / 100, events.at(0).at(image).rows * 54 / 100));		//top right bottomgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 66 / 100, events.at(0).at(image).rows * 9 / 10));		//bottom right bottomgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 66 / 100, events.at(0).at(image).rows-1));				//ground right bottomgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 40 / 100, events.at(0).at(image).rows-1));				//ground left bottom
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 40 / 100, events.at(0).at(image).rows * 9 / 10));		//bottom left bottomgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 22 / 100, events.at(0).at(image).rows * 54 / 100));	//top left bottomgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 37 / 100, events.at(0).at(image).rows * 46 / 100));	//ground left topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 37 / 100, events.at(0).at(image).rows * 37 / 100));	//bottom left topgate
		ROIpoly.push_back(Point(events.at(0).at(image).cols * 24 / 100, 0));	//top left topgate
		polylines(ROIzone, ROIpoly, true, Scalar(200, 200, 200));
		polylines(events.at(0).at(image), ROIpoly, true, Scalar(100, 100, 100));
		
	objectDeleted:;
		for (int object = 0; object < bboxes.size(); object++) {
			Point centerPerson;
			centerPerson.x = bboxes[object].x + bboxes[object].width / 2;
			centerPerson.y = bboxes[object].y + bboxes[object].height / 2;
	
			if (pointPolygonTest(Mat(ROIpoly), centerPerson, true) > 0) {
				//Point in ROI
				rectangle(events.at(0).at(image), centerPerson, centerPerson, Scalar(0, 255, 0), 3, 8, 0); // Green point
			}
			else {
				rectangle(events.at(0).at(image), centerPerson, centerPerson, Scalar(0, 0, 255), 3, 8, 0); // Red point
				bboxes.erase(bboxes.begin() + object);
				confidences.erase(confidences.begin() + object);
				classIds.erase(classIds.begin() + object);
				cout << "Person detected outside ROI" << endl;
				goto objectDeleted;
			}
		}


		//Do non maximum suppression
		vector<int> indices;
		NMSBoxes(bboxes, confidences, confThreshold, nmsThreshold, indices);

		//Store persons
		vector<Rect> persons;
		vector<float> confPersons;
		vector<int> classPersons;
		for (unsigned i = 0; i < indices.size(); i++) {
			int idx = indices[i];
			persons.push_back(bboxes[idx]);
			confPersons.push_back(confidences[idx]);
			classPersons.push_back(classIds[idx]);
		}

		/*
		//Get X-locations
		vector<int>Xpos;
		for (unsigned i = 0; i < persons.size(); i++) {
			Xpos.push_back(persons.at(i).x);
		}
		sort(Xpos.begin(), Xpos.end());

		///cout << "\nTotal persons: " << Xpos.size() << "." << endl;

		//filter top/bottom (too close = 1 person)
		for (unsigned i = 1; i < Xpos.size(); i++) {
			if (Xpos[i] - Xpos[i - 1] <= 0.25 * events.at(0).at(image).cols) {
				Xpos.erase(Xpos.begin() + i);
			}
		}

		///cout << "Too close filtered out : " << Xpos.size() << " persons." << endl;


		//Check if new persons, known person or persons out of view
		vector<int>newKnownXpos;
		vector<int>newCountXpos;
		int k =0;
		///cout << "known Xpos: " << knownXpos.size();
		for (unsigned i = 0; i < knownXpos.size(); i++) {
			///cout << "\n\tcount: " << countXpos[i];
			if (k < Xpos.size()) {
				if (Xpos[k] <= knownXpos[i]+ (0.15* events.at(0).at(image).cols)) {
					//track
					///cout << "\ti=" << i << ", tracked" << endl;;
					newKnownXpos.push_back(Xpos[k]);
					newCountXpos.push_back(0);
					k++; //goto next person
				}
				else {
					//person out of view, increase counter
					///cout << "\ti=" << i << ", counter++";
					newKnownXpos.push_back(knownXpos[i]);
					newCountXpos.push_back(countXpos[i]+1);
					///cout << "\tnew count: " << newCountXpos.at(newCountXpos.size()-1) << endl;
				}
			}
			else {
				///cout << "\ti=" << i << ", counterv2++";
				newKnownXpos.push_back(knownXpos[i]);
				newCountXpos.push_back(countXpos[i]+1);
				///cout << "\tnew count: " << newCountXpos.at(newCountXpos.size()-1) << endl;
			}
		}
		///cout << endl;
		if (k < Xpos.size()) {
			//new persons
			///cout << "New person in view" << endl;
			newKnownXpos.push_back(Xpos[k]);
			newCountXpos.push_back(0);
		}
		///cout << "new Xpos: " << newKnownXpos.size() << "." << endl;

		for (unsigned i = 0; i < newCountXpos.size(); i++) {
			if (newCountXpos[i] > 1) {
				//ADD PERSON PASSED
				newKnownXpos.erase(newKnownXpos.begin() + i);
				newCountXpos.erase(newCountXpos.begin() + i);
				personsPassedV2++;
				///cout << "Add person" << endl;
			}
		}

		knownXpos = newKnownXpos;
		newKnownXpos.clear();
		countXpos = newCountXpos;
		newCountXpos.clear();
		*/

		//Draw rectangles and count people
		int personsCam1 = 0, personsCam2 = 0;
		for (unsigned i = 0; i < persons.size(); i++) {
			double green = confPersons[i] * 2 - 1;
			double red = 1 - (confPersons[i] * 2 - 1);
			Scalar color = Scalar(0, green * 255, red * 255);
			rectangle(events.at(0).at(image), persons[i], color, 2);
			if (persons[i].y + persons[i].height / 2 < events.at(0).at(image).rows / 2) {
				personsCam1++;
			}
			else {
				personsCam2++;
			}
			///cout << classes[classPersons[i]] << " _ ";
		}

		personsInPoort = max(personsCam1, personsCam2);
		if (personsInPoort < pervPersonsInPoort) {
			personsPassed += (pervPersonsInPoort - personsInPoort);
		}


		pervPersonsInPoort = personsInPoort;


		//get string with date (YY-MM-DD) and time (HH:MM) and total event number and snapshot number "19-06-06_10:16_4_1.jpg"
		time_t rawtime;
		struct tm timeinfo;
		char buffer[80];
		time(&rawtime);
		localtime_s(&timeinfo, &rawtime);
		strftime(buffer, sizeof(buffer), "%d-%m-%Y_%H-%M", &timeinfo);

		string inout = "In";
		if (inOutEvents[0] == false) { inout = "Out"; }

		string outputDir = "C:/Users/Jan-Willem/Nextcloud/development/stages/IR_beams_slimme_poort_alex/photos_demo_slimmepoort/";
		stringstream outputFolder, outputFilename;
		outputFolder << outputDir << "Event " << std::setw(3) << std::setfill('0') << eventCounter << " " << inout;




		//mkdir("C:/Users/Jan-Willem/Nextcloud/development/stages/IR_beams_slimme_poort_alex/photos_demo_slimmepoort/event" + eventCounter);
		if (CreateDirectory(outputFolder.str().c_str(), NULL) ||
			ERROR_ALREADY_EXISTS == GetLastError())
		{
			outputFilename << outputFolder.str() << "/" << buffer << "__event-" << eventCounter << "__snapshot-" << snapCounter << "__detection-" << personsInPoort << ".jpg";
			string outputFile;
			outputFile = outputFilename.str();
			cout << outputFile << endl;
			imwrite(outputFile, events.at(0).at(image));
		}
		else
		{
			// Failed to create directory.
			cout << "Error: Failed to create output folder" << endl;
		}
		snapCounter++;
	}

	eventCounter++;

	///personsPassedV2 += knownXpos.size();
	personsPassed += personsInPoort;

	if (inOutEvents[0] == true) {
		//in
		personsInRoom += personsPassed;
		cout << "Total persons in = " << personsInRoom << " (+" << personsPassed << ")" << endl;

	}
	else if (inOutEvents[0] == false) {
		//out
		personsInRoom -= personsPassed;
		cout << "Total persons in = " << personsInRoom << " (-" << personsPassed << ")" << endl;

	}

	events.erase(events.begin());
	inOutEvents.erase(inOutEvents.begin());
	destroyAllWindows();
	goto processNextEvent;

	return;
}
