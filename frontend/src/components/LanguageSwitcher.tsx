import React from 'react';
import { useI18n } from '../i18n/I18nContext';

const LanguageSwitcher: React.FC = () => {
  const { language, setLanguage } = useI18n();

  return (
    <div className="flex gap-2">
      <button
        onClick={() => setLanguage("en")}
        className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          language === "en"
            ? "bg-blue-600 text-white"
            : "bg-slate-800 text-slate-300 hover:bg-slate-700"
        }`}
      >
        EN
      </button>
      <button
        onClick={() => setLanguage("fa")}
        className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
          language === "fa"
            ? "bg-blue-600 text-white"
            : "bg-slate-800 text-slate-300 hover:bg-slate-700"
        }`}
      >
        FA
      </button>
    </div>
  );
};

export default LanguageSwitcher;

