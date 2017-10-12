//Config.h
#pragma once

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

using namespace std;


/*
* \brief Generic configuration Class
*
*/
class Config {
protected:
	string m_Delimiter;  //!< separator between key and value
	string m_Comment;    //!< separator between value and comments
	map<string,std::string> m_Contents;  //!< extracted keys and values
    map<string, string>

	typedef map<string,string>::iterator mapi;
	typedef map<string,string>::const_iterator mapci;
    istream file_in;
public:

	Config( std::string filename,std::string delimiter = "=",std::string comment = "#" );
    Config ();
    template <class T> bool Read (string in_key, T &in_value, string type);

	template<class T> T Read( const std::string& in_key ) const;  //!<Search for key and read value or optional default value, call as read<T>
	template<class T> T Read( const std::string& in_key, const T& in_value ) const;
	template<class T> bool ReadInto( T& out_var, const std::string& in_key ) const;
	template<class T>
	bool ReadInto( T& out_var, const std::string& in_key, const T& in_value ) const;
	void ReadFile(std::string filename,std::string delimiter = "=",std::string comment = "#" );

};


template<class T>
T Config::Read( const std::string& key ) const
{
	// Read the value corresponding to key
	mapci p = m_Contents.find(key);
	if( p == m_Contents.end() ) throw Key_not_found(key);
	return string_as_T<T>( p->second );
}


template<class T>
T Config::Read( const std::string& key, const T& value ) const
{
	// Return the value corresponding to key or given default value
	// if key is not found
	mapci p = m_Contents.find(key);
	if( p == m_Contents.end() ) return value;
	return string_as_T<T>( p->second );
}


template<class T>
bool Config::ReadInto( T& var, const std::string& key ) const
{
	// Get the value corresponding to key and store in var
	// Return true if key is found
	// Otherwise leave var untouched
	mapci p = m_Contents.find(key);
	bool found = ( p != m_Contents.end() );
	if( found ) var = string_as_T<T>( p->second );
	return found;
}


template<class T>
bool Config::ReadInto( T& var, const std::string& key, const T& value ) const
{
	// Get the value corresponding to key and store in var
	// Return true if key is found
	// Otherwise set var to given default
	mapci p = m_Contents.find(key);
	bool found = ( p != m_Contents.end() );
	if( found )
		var = string_as_T<T>( p->second );
	else
		var = value;
	return found;
}


