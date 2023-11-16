import { useState } from "react";
import { downloadResults, downloadZipResults } from "../services/download";
import download from 'downloadjs';
import ShowCSVData from "./ShowCSVData";

export default function Details() {

    const [option, setOption] = useState('one');
    const [showRLG, setShowRLG] = useState(false);
    const [RLG, setRLG] = useState(null);
    const [RLF, setRLF] = useState(null);

    function changeOption({value}) {
        setOption(value);
    }

    async function showRLGText() {
        setShowRLG(!showRLG);
        if(!RLG) {
            const response = await downloadResults('RLG');
            if(response.status != 404) setRLG(response.data);
            console.log(response);
        }
    }

    async function showRLFData() {
        setOption("two");
        const response = await downloadResults('RLF');
        if(response.status != 404) setRLF(response.data);
    }

    async function downloadRLFData(type: string) {
      if (type === 'RLF') {
        const response = await downloadResults(type);
        if (response.status != 404 && response.data!== null) {
          download(response.data, "RLF_Reports.csv");
        }
      } else {
        try {
          const response = await downloadZipResults(type);
          const blob = new Blob([response], { type: 'application/zip' });
    
          const url = window.URL.createObjectURL(blob);
    
          const a = document.createElement('a');
          a.href = url;
          a.download = 'RLF_Snapshots.zip';
          a.click();
    
          window.URL.revokeObjectURL(url);
        } catch (error) {
          console.error('Error downloading zip file:', error);
        }
      }
    }

    async function downloadFullData() {
      try {
        const response = await downloadZipResults('full');
        const blob = new Blob([response], { type: 'application/zip' });
  
        const url = window.URL.createObjectURL(blob);
  
        const a = document.createElement('a');
        a.href = url;
        a.download = 'RLF_Report.zip';
        a.click();
  
        window.URL.revokeObjectURL(url);
      } catch (error) {
        console.error('Error downloading zip file:', error);
      }
    }

  return (
    <>
      <div className="w-3/5 bg-white border border-gray-200 rounded-lg shadow">
        <ul
          className="flex flex-wrap text-sm font-medium text-center text-gray-500 border-b border-gray-200 rounded-t-lg bg-gray-50"
          id="defaultTab"
          data-tabs-toggle="#defaultTabContent"
          role="tablist"
        >
          <li className="me-2">
            <button
              id="about-tab"
              data-tabs-target="#about"
              type="button"
              role="tab"
              aria-controls="about"
              aria-selected="true"
              className={`inline-block p-4 text-lg rounded-ss-lg hover:bg-gray-100 ${option === 'one' ? 'text-primary' : 'text-gray-400'}`}
              onClick={() => changeOption({value: 'one'})}
            >
             1. DOM Extract
            </button>
          </li>
          <li className="me-2">
            <button
              id="services-tab"
              data-tabs-target="#services"
              type="button"
              role="tab"
              aria-controls="services"
              aria-selected="false"
              className={`inline-block p-4 text-lg rounded-ss-lg hover:bg-gray-100 ${option === 'two' ? 'text-primary' : 'text-gray-400'}`}
              onClick={() => showRLFData()}
            >
              2. Failure Detection
            </button>
          </li>
          <li className="me-2">
            <button
              type="button"
              className={`inline-block p-4 text-lg rounded-ss-lg hover:bg-gray-100 ${option === 'three' ? 'text-primary' : 'text-gray-400'}`}
              onClick={() => changeOption({value: 'three'})}
            >
              3. Repair
            </button>
          </li>
        </ul>
        <div id="defaultTabContent">
          <div
            className={`p-4 bg-white rounded-lg md:p-8 ${option === 'one' ? 'block' : 'hidden'}`}
          >
            <h2 className="mb-3 text-3xl font-extrabold tracking-tight text-gray-900">
              Generated Responsive Layout Graph from Extracted DOM
            </h2>
            <p className="mb-3 text-gray-500 text-justify">
            The basis of our RLF detection process is a model of a page’s responsive layout which is called Responsive Layout Graph (RLG). An RLG is automatically obtained by querying the DOM
            of a web page to find the HTML elements involved in the page, and their co-ordinates, at different viewport widths. RLG organizes this information to track the dynamic visibility 
            and relative alignment of these HTML elements as the layout of the page adjusts in relation to viewport width, in accordance with its responsive design.
            </p>
            { !showRLG && <p className="text-primary font-semibold cursor-pointer mb-4" onClick={() => showRLGText()}>View generated RLG</p> }
            { showRLG && <p className="text-primary font-semibold cursor-pointer mb-4" onClick={() => showRLGText()}>Hide generated RLG</p> }
            {
                showRLG && RLG ? (<p className="whitespace-pre overflow-x-auto">{RLG}</p>) : (<p>Loading...</p>)
            }
          </div>
          <div
            className={`p-4 bg-white rounded-lg md:p-8 ${option === 'two' ? 'block' : 'hidden'}`}
          >
            <h2 className="mb-3 text-3xl font-extrabold tracking-tight text-gray-900">
              Responsive Layout Failure Detection
            </h2>
            <p className="mb-3 text-gray-500 dark:text-gray-400 text-justify">
            In the second step, the tool identifies whether a failure exists, which HTML elements were involved and at which particular viewport widths, to help developers diagnose the fault.
            It outputs a set of one or more reports for the webpage and its intended responsive range. Each report refers to a specific detected RLF, comprising its type; details about 
            the HTML elements involved in the RLF; and the RLF range, {'{'} failmin. .failmax {'}'}, or the set of viewports in which the layout failure occurs.
            </p>
            { RLF ? <div className="overflow-x-auto"><ShowCSVData csvString={RLF}/></div> : <p>Loading RLF data...</p> }
            <p className="text-primary font-semibold cursor-pointer my-2" onClick={() => downloadRLFData('RLF')}>Download RLF Reports</p>
            <p className="mb-3 text-gray-500 dark:text-gray-400 text-justify">
            If the generated report shows an evident RLF (i.e., a true positive (TP)) then the developer can debug and fix the issue. If no failure is visible (i.e., a false positive (FP)) then no action
            is required. The reason is, DOM based RLF Detection is prone to non-observable issues — issues that are apparent in the DOM but which are not visibly evident in the page itself. For instance, 
            two HTML elements may overlap, as discovered from checking the coordinates of each element in the DOM, but if they are both transparent and the content of one does not 
            overwrite that of the other, the collision of elements will not be visible to a human looking at the page.
            </p>
            <h1 className='text-center font-title font-bold mb-4'>Generated Snapshots of RLF Regions</h1>
            <p className="mb-3 text-gray-500 dark:text-gray-400 text-justify">
                The tool also outputs a set of snapshots of the webpage at the viewport widths in which the RLFs occur, to help developers visualize the RLFs and their context. The failure
                region is marked with red borders in the snapshots.
            </p>
            <p className="text-primary font-semibold cursor-pointer my-2" onClick={() => downloadRLFData('RLF_snapshot')}>Download RLF Snapshots</p>
            <button className="px-2 py-1 mt-4 bg-primary text-white rounded-md" onClick={() => downloadFullData()}>Download reports as a zip</button>
          </div>
          <div
            className={`p-4 bg-white rounded-lg md:p-8 ${option === 'three' ? 'block' : 'hidden'}`}
          >
            <h2 className="mb-3 text-3xl font-extrabold tracking-tight text-gray-900">
              Repair Generation for RLFs
            </h2>
            <p className="mb-3 text-gray-500 dark:text-gray-400 text-justify">
            The tool has repaired the RLF(s), identified by the previous step. The four stages of repair, are Patch Sourcing, Patch Generation, Patch Injection, and Repair Confirmation.
            
            </p>

          </div>
        </div>
      </div>
    </>
  );
}
