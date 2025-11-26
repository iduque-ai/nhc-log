
import React, { useMemo } from 'react';
import { FilterState, LogLevel, FileInfo } from '../types.ts';
import { CustomMultiSelect } from './CustomMultiSelect.tsx';
import { formatFileSize } from '../utils/helpers.ts';

interface SidebarProps {
  filterState: FilterState;
  fixedFilters?: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
  allDaemons: string[];
  allModules: string[];
  allFunctionNames: string[];
  onNewTab: () => void;
  isLoading: boolean;
  fileInfos: FileInfo[];
  onAppendFiles: (files: File[]) => void;
  onRemoveFile: (fileId: string) => void;
  selectedTimezone: string;
  setSelectedTimezone: (value: string) => void;
  filtersDisabled?: boolean;
  currentKeywordInput: string;
  onCurrentKeywordInputChange: (value: string) => void;
}

const logLevels = Object.values(LogLevel);

const parseDateInTimezone = (dateString: string, timezone: string): Date | null => {
    if (!dateString) return null;

    if (timezone === 'local') {
        return new Date(dateString);
    }
    if (timezone === 'UTC') {
        return new Date(`${dateString}Z`);
    }

    try {
        // For IANA timezones, we construct an ISO string with the correct offset.
        // 1. Create a stable date object by interpreting input as UTC to avoid local timezone biases.
        const [datePart, timePart] = dateString.split('T');
        if (!datePart || !timePart) return new Date(dateString); // Fallback for unexpected format
        
        const [year, month, day] = datePart.split('-').map(Number);
        const [hour, minute, second] = timePart.split(':').map(Number);
        
        // This date represents the point in time if the user input were UTC. It's a stable reference.
        const sampleDate = new Date(Date.UTC(year, month - 1, day, hour, minute, second));

        // 2. Find the offset for that time in the target timezone.
        const formatter = new Intl.DateTimeFormat('en-US', {
            timeZone: timezone,
            timeZoneName: 'longOffset',
        });
        const parts = formatter.formatToParts(sampleDate);
        const offsetPart = parts.find(p => p.type === 'timeZoneName');
        
        if (offsetPart) {
            // e.g., 'GMT-4' or 'GMT+5:30' -> '-04:00' or '+05:30'
            const offsetString = offsetPart.value.replace('GMT', '');
            const sign = offsetString.startsWith('-') ? '-' : '+';
            const numParts = offsetString.substring(1).split(':');
            const hours = numParts[0].padStart(2, '0');
            const minutes = (numParts[1] || '00').padEnd(2, '0');
            const finalOffset = `${sign}${hours}:${minutes}`;
            
            return new Date(`${dateString}${finalOffset}`);
        }
    } catch (e) {
        console.error(`Error parsing date in timezone "${timezone}":`, e);
    }
    
    // Fallback for invalid timezone or other errors
    return new Date(dateString);
};

const formatDateForDateTimeLocalInTimezone = (date: Date | null, timezone: string): string => {
    if (!date) return '';

    let options: Intl.DateTimeFormatOptions = {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
    };

    if (timezone !== 'local') {
        try {
            new Intl.DateTimeFormat(undefined, { timeZone: timezone });
            options.timeZone = timezone;
        } catch (e) {
            console.warn(`Invalid timezone "${timezone}", falling back to local.`);
        }
    }

    // 'sv-SE' (Swedish) locale gives a clean YYYY-MM-DD HH:MM:SS format
    const formatted = new Intl.DateTimeFormat('sv-SE', options).format(date);
    return formatted.replace(' ', 'T');
};


interface KeywordFilterProps {
  queries: string[];
  fixedQueries?: string[];
  onChange: (queries: string[]) => void;
  disabled: boolean;
  highlightEnabled: boolean;
  onHighlightChange: (enabled: boolean) => void;
  inputValue: string;
  onInputValueChange: (value: string) => void;
}

const KeywordFilter: React.FC<KeywordFilterProps> = ({ 
  queries, 
  fixedQueries = [],
  onChange, 
  disabled, 
  highlightEnabled, 
  onHighlightChange,
  inputValue,
  onInputValueChange
}) => {
  const handleAddQuery = () => {
    const trimmedQuery = inputValue.trim();
    if (trimmedQuery && !queries.includes(trimmedQuery)) {
      onChange([...queries, trimmedQuery]);
      onInputValueChange('');
    }
  };

  const handleRemoveQuery = (queryToRemove: string) => {
    if (fixedQueries.includes(queryToRemove)) return;
    onChange(queries.filter(q => q !== queryToRemove));
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAddQuery();
    }
  };
  
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
          <label htmlFor="keyword" className="block text-sm font-medium text-gray-400">Keyword Search</label>
          <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-500" id="highlight-label">Highlight</span>
              <button
                type="button"
                role="switch"
                aria-checked={highlightEnabled}
                aria-labelledby="highlight-label"
                onClick={() => !disabled && onHighlightChange(!highlightEnabled)}
                className={`${
                  highlightEnabled ? 'bg-blue-600' : 'bg-gray-600'
                } relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50`}
                disabled={disabled}
              >
                <span
                  aria-hidden="true"
                  className={`${
                    highlightEnabled ? 'translate-x-4' : 'translate-x-0'
                  } pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
                />
              </button>
          </div>
      </div>
      <div className="flex space-x-2">
        <input
          type="text"
          id="keyword"
          value={inputValue}
          onChange={(e) => onInputValueChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g. (error || fail)"
          disabled={disabled}
          className="flex-grow w-full bg-gray-700 border border-gray-600 text-white rounded-md shadow-sm p-2 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-800"
        />
        <button
          onClick={handleAddQuery}
          disabled={disabled || !inputValue.trim()}
          className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors text-sm font-medium disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {queries.map((q) => {
          const isFixed = fixedQueries.includes(q);
          return (
            <div key={q} className={`flex items-center text-sm font-medium pl-2 pr-1 py-1 rounded-full ${isFixed ? 'bg-gray-700 text-gray-400 cursor-default' : 'bg-gray-600 text-gray-200'}`}>
              <span className="max-w-xs truncate" title={q}>{q}</span>
              {!isFixed && (
                <button
                  onClick={() => !disabled && handleRemoveQuery(q)}
                  disabled={disabled}
                  className="ml-1 text-gray-400 hover:text-white rounded-full focus:outline-none focus:ring-1 focus:ring-white disabled:opacity-50"
                  aria-label={`Remove filter: ${q}`}
                >
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" /></svg>
                </button>
              )}
            </div>
          )
        })}
      </div>
      <p className="text-xs text-gray-500 mt-1">
        Each filter is combined with AND. Use &&, ||, !, () within a filter.
      </p>
    </div>
  );
};

const DataSources: React.FC<{
  fileInfos: FileInfo[];
  onAppendFiles: (files: File[]) => void;
  onRemoveFile: (fileId: string) => void;
  disabled: boolean;
}> = ({ fileInfos, onAppendFiles, onRemoveFile, disabled }) => {
  const [isOpen, setIsOpen] = React.useState(true);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleAddFilesClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onAppendFiles(Array.from(e.target.files));
      e.target.value = ''; // Reset file input
    }
  };

  return (
    <div className="border-t border-b border-gray-700 py-4">
      <button onClick={() => setIsOpen(!isOpen)} className="w-full flex justify-between items-center text-lg font-bold text-white mb-2">
        <span>Data Sources</span>
        <svg className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
      </button>
      {isOpen && (
        <div className="space-y-2">
          <div className="max-h-40 overflow-y-auto space-y-1 pr-2">
              {fileInfos.map(file => (
                <div key={file.id} className="flex justify-between items-center bg-gray-800 p-2 rounded-md">
                  <div className="text-sm">
                    <p className="font-medium text-gray-300 truncate" title={file.name}>{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                  <button onClick={() => onRemoveFile(file.id)} disabled={disabled} className="text-gray-500 hover:text-red-400 disabled:opacity-50 disabled:cursor-not-allowed">&times;</button>
                </div>
              ))}
          </div>
          <button
            onClick={handleAddFilesClick}
            disabled={disabled}
            className="w-full bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors text-sm disabled:bg-gray-600 disabled:cursor-not-allowed"
          >
            {fileInfos.length > 0 ? 'Add More Files' : 'Add Files'}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            multiple
            onChange={handleFileChange}
            disabled={disabled}
            accept=".txt,.log,.zip,.gz"
          />
        </div>
      )}
    </div>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({
  filterState,
  fixedFilters,
  onFilterChange,
  allDaemons,
  allModules,
  allFunctionNames,
  onNewTab,
  isLoading,
  fileInfos,
  onAppendFiles,
  onRemoveFile,
  selectedTimezone,
  setSelectedTimezone,
  currentKeywordInput,
  onCurrentKeywordInputChange,
  filtersDisabled = false,
}) => {
  const { selectedLevels, selectedDaemons, selectedModules, selectedFunctionNames, dateRange, keywordQueries, enableKeywordHighlight } = filterState;
  const filtersDisabledEffective = filtersDisabled || isLoading;

  const handleDateChange = (index: 0 | 1, value: string) => {
    const newDateRange = [...dateRange] as [Date | null, Date | null];
    newDateRange[index] = value ? parseDateInTimezone(value, selectedTimezone) : null;
    onFilterChange({ dateRange: newDateRange });
  };
  
  const uniqueModules = useMemo(() => {
    if (fixedFilters?.selectedModules && fixedFilters.selectedModules.length > 0) {
      return allModules.filter(m => fixedFilters.selectedModules?.includes(m));
    }
    return allModules;
  }, [allModules, fixedFilters]);
  
  const uniqueFunctionNames = useMemo(() => {
    if (fixedFilters?.selectedFunctionNames && fixedFilters.selectedFunctionNames.length > 0) {
      return allFunctionNames.filter(m => fixedFilters.selectedFunctionNames?.includes(m));
    }
    return allFunctionNames;
  }, [allFunctionNames, fixedFilters]);

  return (
    <aside className="w-80 bg-gray-800 flex flex-col p-4 border-r border-gray-700 h-full">
      <h1 className="text-xl font-bold text-white mb-4">NHC Log Viewer</h1>

      <DataSources
        fileInfos={fileInfos}
        onAppendFiles={onAppendFiles}
        onRemoveFile={onRemoveFile}
        disabled={isLoading}
      />

      <div className="flex-grow space-y-4 pt-4 overflow-y-auto min-h-0">
        <h2 className="text-lg font-bold text-white mb-2">Filters</h2>
        
        <KeywordFilter
          queries={keywordQueries}
          fixedQueries={fixedFilters?.keywordQueries}
          onChange={(queries) => onFilterChange({ keywordQueries: queries })}
          disabled={filtersDisabledEffective}
          highlightEnabled={enableKeywordHighlight}
          onHighlightChange={(enabled) => onFilterChange({ enableKeywordHighlight: enabled })}
          inputValue={currentKeywordInput}
          onInputValueChange={onCurrentKeywordInputChange}
        />

        <CustomMultiSelect
          label="Log Levels"
          options={logLevels}
          selected={selectedLevels}
          onChange={(selected) => onFilterChange({ selectedLevels: selected as LogLevel[] })}
          disabled={filtersDisabledEffective}
          fixedSelected={fixedFilters?.selectedLevels}
        />
        
        <CustomMultiSelect
          label="Daemons"
          options={allDaemons}
          selected={selectedDaemons}
          onChange={(selected) => onFilterChange({ selectedDaemons: selected })}
          disabled={filtersDisabledEffective}
          fixedSelected={fixedFilters?.selectedDaemons}
        />
        
        <CustomMultiSelect
          label="Modules"
          options={uniqueModules}
          selected={selectedModules}
          onChange={(selected) => onFilterChange({ selectedModules: selected })}
          disabled={filtersDisabledEffective}
          fixedSelected={fixedFilters?.selectedModules}
        />
        
        <CustomMultiSelect
          label="Function Names"
          options={uniqueFunctionNames}
          selected={selectedFunctionNames}
          onChange={(selected) => onFilterChange({ selectedFunctionNames: selected })}
          disabled={filtersDisabledEffective}
          fixedSelected={fixedFilters?.selectedFunctionNames}
        />

        <div>
            <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-gray-400">Date Range</label>
                <select
                    id="timezone-select"
                    value={selectedTimezone}
                    onChange={(e) => setSelectedTimezone(e.target.value)}
                    disabled={isLoading}
                    className="bg-gray-700 border border-gray-600 text-white rounded-md p-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                    title="Select Timezone"
                >
                    <option value="local">Local Time</option>
                    <option value="UTC">UTC</option>
                    <option value="America/New_York">America/New_York (ET)</option>
                    <option value="America/Chicago">America/Chicago (CT)</option>
                    <option value="America/Denver">America/Denver (MT)</option>
                    <option value="America/Los_Angeles">America/Los_Angeles (PT)</option>
                    <option value="Europe/London">Europe/London (GMT/BST)</option>
                    <option value="Europe/Berlin">Europe/Berlin (CET/CEST)</option>
                    <option value="Asia/Tokyo">Asia/Tokyo (JST)</option>
                </select>
            </div>
            <div className="flex flex-col space-y-2">
                <input 
                  type="datetime-local" 
                  step="1"
                  value={formatDateForDateTimeLocalInTimezone(dateRange[0], selectedTimezone)} 
                  onChange={(e) => handleDateChange(0, e.target.value)} 
                  disabled={filtersDisabledEffective} 
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md p-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" 
                />
                <input 
                  type="datetime-local" 
                  step="1"
                  value={formatDateForDateTimeLocalInTimezone(dateRange[1], selectedTimezone)} 
                  onChange={(e) => handleDateChange(1, e.target.value)} 
                  disabled={filtersDisabledEffective} 
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md p-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500" 
                />
            </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700 space-y-4">
        <button
          onClick={onNewTab}
          disabled={isLoading}
          className="w-full bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors text-sm font-medium disabled:bg-gray-600 disabled:cursor-not-allowed"
        >
          {filtersDisabled ? 'Create New Tab' : 'Create New Tab from Filters'}
        </button>
      </div>
    </aside>
  );
};
