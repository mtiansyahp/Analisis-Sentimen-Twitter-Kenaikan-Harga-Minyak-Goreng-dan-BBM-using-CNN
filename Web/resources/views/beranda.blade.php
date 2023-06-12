<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    @vite('resources/css/app.css')
    <title>Laravel</title>

    <!-- Fonts -->
    <link href="https://fonts.bunny.net/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">

    <!-- Styles -->

</head>

<body class="antialiased h-full">
    <div class="min-h-full">
        <nav class="bg-gray-800">
            <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                <div class="flex h-16 items-center justify-between">
                    <div class="flex items-center">
                        <div class="flex-shrink-0">
                            <img class="h-8 w-8" src="https://tailwindui.com/img/logos/mark.svg?color=indigo&shade=500"
                                alt="Your Company">
                        </div>
                        <div class="hidden sm:block">
                            <div class="ml-10 flex items-baseline space-x-4">
                                <!-- Current: "bg-gray-900 text-white", Default: "text-gray-300 hover:bg-gray-700 hover:text-white" -->
                                <a href="#" class="bg-gray-900 text-white px-3 py-2 rounded-md text-sm font-medium"
                                    aria-current="page">Beranda</a>
                                {{-- <a href="#"
                                    class="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Dokumentasi</a>
                                --}}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </nav>

        <header class="bg-white shadow">
            <div class="mx-auto max-w-7xl py-6 px-4 sm:px-6 lg:px-8">
                <h1 class="text-3xl font-bold tracking-tight text-gray-900">Sistem Analisis Sentimen Twitter</h1>
                <h2 class="text-sm font-text-regular tracking-tight my-2">Pesatnya perkembangan media sosial membuat
                    pengguna lebih mudah membagikan pendapat pada platform online seperti forum web, blog, dan situs
                    media sosial Twitter maupun Facebook (Rani dan Kumar, 2018; Zhao et al., 2019). Berdasarkan data
                    dari Asosiasi Penyelenggara Jasa Internet Indonesia (APJII), pertumbuhan pemakai internet tahun 2022
                    (Q1), pengguna internet di Indonesia mencapai 220 juta dari total keseluruhan populasi penduduk
                    Indonesia. Sebelum pandemi, jumlah pengguna internet di Indonesia adalah 175 juta orang, sedangkan
                    pada 2020-2021 naik di angka 210,03 juta dari 196,7 juta pada periode 2019-2020 (APJII, 2021).
                    Mayoritas layanan internet yang paling sering digunakan antara lain seperti Facebook, Twitter,
                    Instagram dan lainnya. (APJII, 2022). </h2>
            </div>
        </header>
        <main>
            <div class="mx-auto max-w-7xl py-6 sm:px-6 lg:px-8">
                <div class="bg-red-500">
                    @if ($errors->any())
                    <div class="mx-auto max-w-7xl py-3 px-3 sm:px-6 lg:px-8">
                        <div class="flex flex-wrap items-center justify-between">
                            <div class="flex w-0 flex-1 items-center">
                                <span class="flex rounded-lg bg-red-500 p-2">
                                    <!-- Heroicon name: outline/megaphone -->
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                                        stroke-width="1.5" stroke="white" class="w-6 h-6">
                                        <path stroke-linecap="round" stroke-linejoin="round"
                                            d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                                    </svg>
                                </span>
                                @foreach ($errors->all() as $error)
                                <p class="ml-3 truncate font-medium text-white">
                                    <span class="hidden md:inline"></span>{{ $error }}
                                </p>
                                @endforeach
                            </div>

                            <div class="order-2 flex-shrink-0 sm:order-3 sm:ml-3">
                                <button type="button"
                                    class="-mr-1 flex rounded-md p-2 hover:bg-red-500 focus:outline-none focus:ring-2 focus:ring-white sm:-mr-2">
                                    <span class="sr-only">Dismiss</span>
                                    <!-- Heroicon name: outline/x-mark -->
                                    <svg class="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none"
                                        viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true">
                                        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>
                    @endif
                </div>

                <!-- Replace with your content -->
                <div class="px-4 py-6 sm:px-0">
                    <form action="{{ route('beranda_post') }}" method="POST" enctype="multipart/form-data">
                        <div class="max-h-96 rounded-lg border-4 border-dashed border-gray-200 pb-48">
                            @csrf
                            <div class="form-1">
                                <div class="mx-3 my-3 md:grid md:grid-cols-3 md:gap-6">
                                    <div class="md:col-span-1">
                                        <div class="px-4 sm:px-0">
                                            <h3 class="text-lg font-medium leading-6 text-gray-900">Prediksi Sentimen
                                            </h3>
                                            <p class="mt-1 text-sm text-gray-600">Masukan Dataset Twitter yang ingin
                                                anda Prediksi Sentimennya</p>
                                        </div>
                                    </div>
                                    <div class="mt-5 md:col-span-2 md:mt-0">
                                        <div class="space-y-6 bg-white px-4 py-5 sm:p-6">
                                            <div class="grid grid-cols-12 gap-6">
                                                <div class="col-span-12 sm:col-span-12">
                                                    <div>
                                                        <label for="tweet"
                                                            class="block text-sm font-medium leading-6 text-gray-900">Contoh
                                                            Tweet</label>
                                                        <div class="relative mt-2 rounded-md shadow-sm">
                                                            <input type="text" name="tweet" id="tweet"
                                                                class="block w-full rounded-md border-0 py-1.5 pl-7 pr-20 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                placeholder="Masukan Contoh Review" @required(true)>
                                                        </div>
                                                        <div class="result">
                                                            <label for="tweet"
                                                                class="block text-sm font-medium leading-6 text-gray-900">Inputan
                                                                Awal Tweet
                                                            </label>
                                                            <input type="text" name="tweet" id="review"
                                                                class="block w-full rounded-md border-0 py-1.5 pl-7 pr-20 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                placeholder="" value="{{$content->tweet ?? ''}} "
                                                                disabled>
                                                            <label for="tweet"
                                                                class="block text-sm font-medium leading-6 text-gray-900">Hasil
                                                                Pre Processing</label>
                                                            <input type="text" name="tweet" id="review"
                                                                class="block w-full rounded-md border-0 py-1.5 pl-7 pr-20 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                placeholder="" value="{{$content->preproc ?? '' }}"
                                                                disabled>
                                                            {{-- {{ $content->preproc }} --}}
                                                            <label for="result-model"
                                                                class="block text-sm font-medium leading-6 text-gray-900">Hasil
                                                                Modelling</label>
                                                            <input type="text" name="tweet" id="review"
                                                                class="block w-full rounded-md border-0 py-1.5 pl-7 pr-20 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                placeholder="" value="{{$content->result ?? '' }}"
                                                                disabled>
                                                            {{-- {{ $content->preproc }} --}}
                                                        </div>
                                                    </div>

                                                </div>
                                            </div>
                                        </div>
                                        <div class="shadow sm:overflow-hidden sm:rounded-md">
                                        </div>

                                    </div>
                                </div>
                            </div>
                            {{-- <div class="form-2">
                                <div class="mx-3 my-3 md:grid md:grid-cols-3 md:gap-6">
                                    <div class="md:col-span-1">
                                        <div class="px-4 sm:px-0">

                                        </div>
                                    </div>
                                    <div class="mt-0 md:col-span-2 md:mt-0">
                                        <div class="space-y-0 bg-white px-4 py-0 sm:p-6">
                                            <div class="grid grid-cols-12 gap-6">
                                                <div class="col-span-12 sm:col-span-12">
                                                    <label for="review"
                                                        class="block text-sm font-medium leading-6 text-gray-900">Contoh
                                                        Review</label>
                                                    <div class="relative mt-2 rounded-md shadow-sm">
                                                        <div class="border-t border-gray-200 px-4 py-6">
                                                            <!-- Filter section, show/hide based on section state. -->
                                                            <div class="pt-0">
                                                                <div class="space-y-6">
                                                                    <div class="flex items-center">
                                                                        <input id="filter-mobile-color-0" name="color[]"
                                                                            value="white" type="checkbox"
                                                                            class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500">
                                                                        <label for="filter-mobile-color-0"
                                                                            class="ml-3 min-w-0 flex-1 text-gray-500">White</label>
                                                                    </div>

                                                                    <div class="flex items-center">
                                                                        <input id="filter-mobile-color-1" name="color[]"
                                                                            value="beige" type="checkbox"
                                                                            class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500">
                                                                        <label for="filter-mobile-color-1"
                                                                            class="ml-3 min-w-0 flex-1 text-gray-500">Beige</label>
                                                                    </div>

                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="shadow sm:overflow-hidden sm:rounded-md">
                                        </div>

                                    </div>
                                </div>
                            </div>
                            <div class="form-3">
                                <div class="mx-3 my-3 md:grid md:grid-cols-3 md:gap-6">
                                    <div class="md:col-span-1">
                                        <div class="px-4 sm:px-0">
                                            <h3 class="text-lg font-medium leading-6 text-gray-900">Dokumentasi</h3>
                                            <p class="mt-1 text-sm text-gray-600">This information will be displayed
                                                publicly so be careful what you share.</p>
                                        </div>
                                    </div>
                                    <div class="mt-5 md:col-span-2 md:mt-0">
                                        <div class="space-y-6 bg-white px-4 py-5 sm:p-6">
                                            <div class="grid grid-cols-12 gap-6">
                                                <div class="col-span-12 sm:col-span-12">
                                                    <div>
                                                        <label for="review"
                                                            class="block text-sm font-medium leading-6 text-gray-900">Contoh
                                                            Review</label>
                                                        <div class="relative mt-2 rounded-md shadow-sm">
                                                            <input type="text" name="review" id="review"
                                                                class="block w-full rounded-md border-0 py-1.5 pl-7 pr-20 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                placeholder="Masukan Contoh Review">
                                                        </div>
                                                    </div>

                                                </div>
                                            </div>
                                        </div>
                                        <div class="shadow sm:overflow-hidden sm:rounded-md">
                                        </div>

                                    </div>
                                </div>
                            </div> --}}
                        </div>
                        <div class="bg-gray-50 px-4 py-3 text-center sm:px-6">
                            <button type="submit"
                                class="w-full inline-flex justify-center rounded-md border border-transparent bg-gray-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2">Proses</button>
                        </div>


                    </form>
                </div>

            </div>
        </main>
    </div>

</body>

</html>
