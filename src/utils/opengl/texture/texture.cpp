
#include "headers.hpp"
#ifdef USE_GUI

#include <sstream>
#include <iterator>
#include <algorithm>

#include "texture.hpp"
#include "globals.hpp"
#include "log.hpp"
#include "utils.hpp"

bool Texture::_init = false;
std::vector<int> Texture::textureLocations;
std::map<unsigned int, long> Texture::locationsHitMap;
std::vector<std::pair<long, unsigned int> > Texture::reversedHitMap;


Texture::Texture(GLenum textureType_) :
	textureType(textureType_), textureId(0), lastKnownLocation(-1),  mipmap(false)
{
	if(!_init) {
		log_console->errorStream() << "[TEXTURE MANAGER] Texture manager has not been initialized !";
		exit(1);
	}

	glGenTextures(1, &textureId);

	std::stringstream ss;
	ss << "[Texture][id=" << textureId << "]  ";
	logTextureHead = ss.str();
} 

Texture::~Texture() {
	glDeleteTextures(1, &textureId);
}

void Texture::addParameter(const Parameter param) {
	params.push_back(param);
}

void Texture::addParameters(const std::list<Parameter> &paramList) {
	for(std::list<Parameter>::const_iterator it = paramList.begin(); it != paramList.end(); ++it) {
		addParameter(*it);
	}
}

const std::list<Parameter> Texture::getParameters() const {
	return params;
}

//texture must be linked
void Texture::applyParameters() const {
	for(std::list<Parameter>::const_iterator it = params.begin(); it != params.end(); ++it) {
		Parameter p = *it;
		switch(p.type()) {
			
			case(ParamType::F):
				glTexParameterf(textureType, p.paramName(), p.params().f);	
				break;
			case(ParamType::I):
				glTexParameteri(textureType, p.paramName(), p.params().i);	
				break;
			case(ParamType::IV):
				glTexParameteriv(textureType, p.paramName(), p.params().iv);	
				break;
			case(ParamType::FV):
				glTexParameterfv(textureType, p.paramName(), p.params().fv);	
				break;
			default: 
				log_console->errorStream() << "[TEXTURE.CPP] The impossible happened !";
				exit(1);
		}
	}
}

unsigned int Texture::getTextureId() const {
	return textureId;
}

void Texture::generateMipMap() {
	mipmap = true;
}
		
void Texture::init() {

	if(Globals::glMaxCombinedTextureImageUnits == 0) {
		log_console->errorStream() << "[TEXTURE MANAGER Init]  Global vars must be initialized first.";
		exit(1);
	}
	
	log_console->infoStream() << "[Texture Manager Init]  Hardware texture units : " << Globals::glMaxCombinedTextureImageUnits;

	for (int i = 0; i < Globals::glMaxCombinedTextureImageUnits; i++) {
		textureLocations.push_back(-1);
		locationsHitMap[static_cast<unsigned int>(i)] = 0ul;
	}

	_init = true;
}
		

std::vector<unsigned int> Texture::requestTextures(unsigned int nbRequested) {
	
	if(nbRequested > (unsigned int) Globals::glMaxCombinedTextureImageUnits) {
		log_console->errorStream() << "[TEXTURE MANAGER]  Received invalid texture request : " << nbRequested  << " (MAX = " << Globals::glMaxCombinedTextureImageUnits << ") ! Your hardware simply can't handle it ! Go fix your shaders or just buy a tri-SLI of Titan-Z !";
		exit(1);
	}

	std::vector<unsigned int> locations;

	for (unsigned int i = 0; i < textureLocations.size() && nbRequested != 0u; i++) {
		if(textureLocations[i] == -1) {
			locations.push_back(i);			
			nbRequested--;
		}
	}
	
	std::vector<std::pair<long, unsigned int> >::reverse_iterator it = reversedHitMap.rbegin();
	while(nbRequested != 0) {
		
		if(it == reversedHitMap.rend()) {
			log_console->fatalStream() << "[TEXTURE MANAGER] Fatal Error !" << nbRequested;
			exit(1);
		}
	
		if(std::find(locations.begin(), locations.end(), it->second) == locations.end()) {
			locations.push_back(it->second);
			nbRequested--;
		}

		++it;
	}

	Texture::sortHitMap();

	return locations;
}


		
void Texture::sortHitMap() {
	//log_console->debugStream() << "[TEXTURE MANAGER]  Sorting texture hitmap !";
	Texture::reversedHitMap = utils::mapToReversePairVector(Texture::locationsHitMap);
	std::sort(reversedHitMap.begin(), reversedHitMap.end(), compareFunc);
}

void Texture::reportHitMap() {

	log_console->infoStream() << "[TEXTURE MANAGER]  Texture units filling report :";
	
	std::cout << "== HIT MAP ==" << std::endl;
	std::map<unsigned int, long>::iterator it = locationsHitMap.begin();
	while(it != locationsHitMap.end()) {
		std::cout << it->first << "=>" << it->second << " "; 
		++it;
	}
	std::cout << std::endl;
	std::cout << "== Statistics ==" << std::endl;
	std::cout  << "Min hits : "<< reversedHitMap[0].first << std::endl <<
		"Max hits : " << reversedHitMap[reversedHitMap.size()-1].first << std::endl;
}

int Texture::getLastKnownLocation() const {
	return lastKnownLocation;
}

bool Texture::isBinded() const {
	return (lastKnownLocation != -1) && (textureId == (unsigned int)textureLocations[lastKnownLocation]);
}

bool Texture::compareFunc(std::pair<long, unsigned int> a, std::pair<long, unsigned int> b) {
	if(a.first < b.first) {
		return true;
	}
	else if(a.first == b.first) {
		return (a.second > b.second);
	}
	else {
		return false;
	}
}

#endif
