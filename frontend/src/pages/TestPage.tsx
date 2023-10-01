import { useLocation } from 'react-router-dom';
import Navbar from '../layouts/Navbar';
import { useState, useEffect } from 'react';
import { testUrl } from '../services/test';

function parseUrlDomain(url) {
    const name = new URL(url).hostname.replace('www.', '');
    return name;
}

export default function TestPage() {

  const [url, setUrl] = useState('');
  const location = useLocation();

  useEffect(() => {
    const parsedUrl = parseUrlDomain(location.state.url);
    setUrl(parsedUrl);
    fetchTestResults(location.state.url);
  }, []);

  const fetchTestResults = async (url) => {
    const response = await testUrl(url);
    console.log(response);
  };

  return (
    <>
      <Navbar />
      <div className='flex justify-center my-24'>
        <h1 className='font-title text-xl lg:text-xl'>Detecting Failures for <b>{url}</b></h1>
      </div>
    </>    
  );
}